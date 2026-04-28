# GLow Gossip Learning Framework - Full Technical Guideline

This guide explains the GLow Gossip Learning implementation from process startup to end-of-run artifacts, including attack variants.

Scope included:
- Core Gossip Learning pipeline: `main.py`, `hydra_main.py`, `custom_strategies/topology_based_GL.py`, `client.py`, `server.py`, `model.py`, `dataset.py`
- Attack variants: `main_backdoor.py`, `client_backdoor.py`, `client_poison.py`

Scope excluded:
- FedAvg baseline (`FL_hydra_main.py`) as primary workflow
- Centralized baseline (`cnl_pytorch.py`) as primary workflow

---

## 1. High-level architecture

GLow uses Flower simulation with a custom strategy (`topology_based_Avg`) to emulate decentralized gossip behavior on top of centralized orchestration.

Main components:
- Entrypoint script:
  - `main.py` (CLI args)
  - `hydra_main.py` (Hydra config)
  - `main_backdoor.py` (attack-enabled run)
- Strategy:
  - `custom_strategies/topology_based_GL.py` (`topology_based_Avg`)
- Client logic:
  - `client.py` (clean)
  - `client_backdoor.py` (trigger backdoor + ASR)
  - `client_poison.py` (label flipping)
- Data/model/server utilities:
  - `dataset.py`, `model.py`, `server.py`

Design intent:
- A topology defines neighborhood relations per node.
- Each communication round focuses on one selected pool/head (round-robin).
- The selected neighborhood is sampled and aggregated.
- State is tracked as one parameter vector per node (`pool_parameters`).

---

## 2. Node creation and topology semantics

### 2.1 Where nodes come from

Nodes are implicit Flower clients created by `generate_client_fn(...)`.

Flow:
1. Load topology YAML (`num_clients`, `pools`, etc.).
2. Build `vcid = np.arange(num_clients)`.
3. Build one trainloader and validationloader per client index.
4. `generate_client_fn` returns a Flower client factory; each simulation client `cid` maps to one `FlowerClient` instance.

Files:
- `main.py` / `hydra_main.py`
- `client.py`

### 2.2 Topology file meaning

Example fields (see `conf/topologies/ring30+5.yaml`):
- `num_clients`: total simulated nodes.
- `max_num_clients_per_round`: max active participants for resource heuristic.
- `clients_with_no_data`: nodes that exist in graph but get empty local dataset.
- `pools.pX`: node neighborhood list used by the strategy criterion.

Important behavior:
- Strategy samples clients by checking `client.cid in topology[selected_pool]`.
- In practice, each `pX` should contain at least `X` itself (self-inclusion) to guarantee the selected head participates in its own round.

---

## 3. Initialization sequence (startup to round 1)

The following sequence is for `main.py`; `hydra_main.py` is similar except config/output wiring.

1. Parse runtime inputs:
   - `python main.py <conf_file> <run_id> <topology_file>`
2. Load YAML config (`conf/base.yaml` style).
3. Configure logging verbosity (`utils/logging.py`).
4. Resolve output directory (`run_name` + timestamp logic).
5. Load topology YAML and build:
   - `num_clients`
   - `topology` list where `topology[i] = pools["p{i}"]`
6. Build datasets and loaders:
   - `prepare_dataset_iid(...)` for CIFAR or
   - `prepare_dataset_mnist_iid(...)` for MNIST
7. Build Flower client factory (`generate_client_fn`).
8. Build strategy (`topology_based_Avg`) with:
   - topology
   - config callback (`get_on_fit_config`)
   - centralized eval callback (`get_evaluate_fn`)
   - metrics aggregation callbacks
9. Build Flower server/config (`num_rounds`).
10. Resolve simulation resource policy (`num_cpus_per_client`, `num_gpus_per_client`).
11. Optional pretraining phase (see Section 6).
12. Start simulation.

---

## 4. Strategy internals: what is stateful in Gossip Learning

`topology_based_Avg` maintains:
- `client_list`: round-robin ordering of head selection.
- `selected_pool`: current round head id.
- `pool_parameters`: list of Flower `Parameters`, one per node id.
- `pool_losses`, `pool_metrics`: final tracked result per node.

### 4.1 Parameter initialization

`initialize_parameters(...)` behavior:
- If `initial_parameters` is a single Flower `Parameters` object (pretraining path), it broadcasts this same initialization to all pools.
- Else, it requests initial parameters from sampled clients and seeds `pool_parameters[i]` per client.

---

## 5. Full round lifecycle (clean pipeline)

Each round executes the following chain.

### 5.1 Head selection and neighborhood sampling

In `configure_fit(...)`:
1. `selected_pool = client_list[0]`
2. Rotate list: `client_list = np.roll(client_list, -1)`
3. Read neighborhood: `connections = topology[selected_pool]`
4. Sample only clients in `connections` using a Flower `Criterion`.

### 5.2 Fit instructions sent to each sampled client

For each sampled client `c`, strategy creates:
- `FitIns(parameters=pool_parameters[c.cid], config=...)`

Config includes:
- `lr`
- `local_epochs`
- `enable_tqdm`
- `local_train_cid = selected_pool`
- `comm_round = server_round`
- `num_nodes = min_available_clients`

Key detail:
- Each client receives its own stored parameter vector (`pool_parameters[cid]`), not a single global vector.

### 5.3 Client fit behavior

In `client.py`:
- All sampled clients call `set_parameters(...)`.
- Only the selected head trains:
  - condition: `config['local_train_cid'] == self.cid` (or `-1` in FL case)
- Head runs local training (`model.train(...)`) and returns updated parameters.
- Non-head sampled clients return unchanged parameters.

### 5.4 Fit aggregation

In `aggregate_fit(...)`:
- Optional `early_local_train` rule: for first `N` rounds (`N = num_nodes`), non-head client contributions are zeroed (`num_examples = 0`).
- Otherwise, weighted average is computed over sampled results.
- Aggregated parameters overwrite only:
  - `pool_parameters[selected_pool] = aggregated_parameters`

Conceptually, selected head state is updated by averaging neighborhood returns; non-selected pools are unchanged that round.

### 5.5 Evaluation phase

In `configure_evaluate(...)`:
- Uses the same `selected_pool` neighborhood sampling criterion.
- Sends `EvaluateIns(parameters=pool_parameters[cid])` to each sampled client.

In client `evaluate(...)`:
- Client evaluates on its validation loader.
- Returns `loss`, `acc_distr`, `cid` (and ASR in backdoor variant).

In `aggregate_evaluate(...)`:
- Weighted loss averaging via Flower helper.
- Aggregates distributed metrics using callback (`cli_eval_distr_results`).
- Logs heartbeat summaries.

### 5.6 Centralized evaluation callback

`strategy.evaluate(...)` calls server closure from `server.py` (`get_evaluate_fn`):
- Evaluates selected pool parameters on centralized/global test loader.
- Returns `acc_cntrl`.
- Updates `pool_losses[selected_pool]` and `pool_metrics[selected_pool]`.
- On final round, triggers `save_results()`.

---

## 6. Local training internals

### 6.1 Model

Default model path in GL scripts is `LeNet` (`model.py`):
- 3 conv blocks
- FC hidden layer + dropout
- Output layer size = `num_classes`

### 6.2 Optimizer and objective

Client clean training (`client.py`):
- Optimizer: Adam
- Loss: CrossEntropy (`model.train`)
- Device resolved by config (`cpu`/`cuda` style strings)

### 6.3 Dataset partitioning

`dataset.py`:
- Supports CIFAR-10 and MNIST.
- IID splitter orders by class then partitions across clients with data.
- Per-client split into train/validation (`val_ratio`, default 0.1).
- Clients in `clients_with_no_data` receive empty placeholders.

---

## 7. Pretraining and initialization shaping

If `pretraining.enabled: true`:
1. Build centralized pretraining loader from all non-empty client train datasets.
2. `load_or_train_pretrained(...)`:
   - load checkpoint if exists, else train and save
3. Extract ndarray parameters from pretrained model.
4. Optional initialization shaping:
   - Blend with random init using `mix_alpha`:

$$
\theta_{init} = \alpha\,\theta_{pretrain} + (1-\alpha)\,\theta_{random}
$$

   - Optional additive Gaussian noise with std `noise_std`.
5. Set strategy `initial_parameters` to this Flower `Parameters` object.

This directly controls initial convergence behavior.

---

## 8. Weight distribution and aggregation mechanics

### 8.1 Parameter format transformations

- Flower uses serialized `Parameters`.
- Clients convert between state dict and ndarray lists:
  - `set_parameters(...)`
  - `get_parameters(...)`

### 8.2 Aggregation formula

Core weighted averaging concept:

$$
\theta^{agg} = \frac{\sum_i n_i\,\theta_i}{\sum_i n_i}
$$

Where:
- $\theta_i$ = client return parameters
- $n_i$ = `fit_res.num_examples`

### 8.3 Edge handling for zero-example totals

`topology_based_GL.py` guards against `num_examples_total == 0` and skips aggregation.

Repository’s Flower modification (`flwr_lib_modifications/aggregate.py`) also has defensive behavior in `aggregate_inplace`:
- if total examples is zero, uses scaling factor `1.0` to avoid divide-by-zero.

---

## 9. Attack variants

## 9.1 Backdoor path (`main_backdoor.py` + `client_backdoor.py`)

Main differences from clean run:
- Same overall strategy lifecycle.
- Client can be malicious by hardcoded id rule (`self.is_malicious = int(cid) in [1]`).

Backdoor mechanics:
1. Wrap local dataset via `BackdoorDataset`.
2. Poison subset ratio (`BACKDOOR_POISON_RATE`) by:
   - applying 3x3 white trigger patch at bottom-right area
   - forcing label to target class (default 0)
3. Train locally on poisoned loader.
4. Optional model boosting for malicious client:

$$
\theta_{boosted} = \theta_{global} + \beta\,(\theta_{local} - \theta_{global})
$$

where $\beta = \text{BACKDOOR_BOOST_FACTOR}$.

Evaluation includes ASR (`test_asr`):
- Apply trigger to validation samples not originally target class.
- Measure fraction predicted as target class.
- Returned metric key: `asr`.

### 9.2 Poisoning path (`client_poison.py`)

Alternative attack client:
- Label flipping via `PoisonedDataset` (`from_class` -> `to_class`).
- Malicious client id is hardcoded (`int(cid) in [2]`).
- No trigger/ASR by default; still uses normal fit/evaluate lifecycle.

---

## 10. Outputs, logs, and artifacts

## 10.1 Logging model

`utils/logging.py` supports levels:
- `minimal`
- `standard`
- `verbose`

Config controls:
- Global: `verbose_logging`
- Per-component overrides: `log_level_client_training`, `log_level_pretraining`, etc.

## 10.2 Clean run artifacts

From `main.py` and strategy finalization:
- `<run_id>_raw.out`
  - distributed/centralized losses
  - distributed metrics arrays (`acc_distr`, `cid`, optional `asr`)
  - centralized metrics (`acc_cntrl`)
  - execution time
- `<run_id>_acc_distr.out`
  - per-round distributed accuracy rows
- `<run_id>_pool.out`
  - final per-pool neighbor/loss/accuracy summary
- `parameters/<pool_id>.pth`
  - saved model states per pool index

## 10.3 Backdoor extra artifact

`main_backdoor.py` also writes:
- `run_summary.json`
  - config snapshot + final metrics + output file names

---

## 11. Configuration reference (what actually affects behavior)

From `conf/base.yaml` and runtime use:

Core run:
- `run_name`
- `topology`
- `device`
- `num_rounds`
- `seed`
- `dataset` (`cifar` or `mnist`)
- `data_path` (dataset download path)

Data and model:
- `num_classes`
- `batch_size`

Training config (`config_fit`):
- `lr`
- `local_epochs`
- `enable_tqdm`

Strategy behavior:
- `early_local_train`

Simulation resources (`simulation`):
- `num_cpus_per_client`
- `num_gpus_per_client` (`auto` or float)

Pretraining (`pretraining`):
- `enabled`
- `epochs`
- `lr`
- `save_path`
- `enable_tqdm`
- `mix_alpha`
- `noise_std`

Topology-dependent controls:
- `num_clients`
- `max_num_clients_per_round`
- `clients_with_no_data`
- `pools.pX`

---

## 12. Worked round example (conceptual)

Assume 3-node ring-like neighborhood definitions:
- `p0 = [0,1]`
- `p1 = [1,2]`
- `p2 = [2,0]`

Suppose round selects `selected_pool = 1`.

1. Sample clients from `p1` => clients {1,2}.
2. Send fit instructions:
   - client 1 gets `pool_parameters[1]`
   - client 2 gets `pool_parameters[2]`
3. Client 1 trains locally and returns updated params.
4. Client 2 returns unchanged params (non-head in clean mode).
5. Aggregate weighted returns to produce `theta_agg`.
6. Set only `pool_parameters[1] = theta_agg`.
7. Evaluate neighborhood and centralized test using selected pool state.
8. Next round rotates head.

---

## 13. Implementation caveats and practical notes

1. Aggregation weights currently use `len(trainloader)` in client returns.
- This is number of batches, not exact sample count.
- It still gives proportional weighting if batch sizes are uniform, but is semantically different from true example count.

2. `clients_with_no_data` need careful handling.
- Empty loaders are represented by placeholders in dataset prep.
- Topology can still include those clients, affecting communication behavior.

3. `save_results()` parameter export currently loads parameters from `self.pool_parameters[self.selected_pool]` inside a loop over all `cli_ID`.
- This means all saved model files may reflect the selected pool snapshot at save time rather than each pool’s own parameters.
- If you need strict per-pool exports, this part should be reviewed.

4. Backdoor/poison malicious client selection is hardcoded in client files.
- To run systematic experiments, move malicious IDs to config.

5. Topology self-inclusion is effectively assumed.
- If a pool does not include itself in `pX`, selected head may not participate in its own fit/eval neighborhood.

---

## 14. End-to-end call-path map

Clean run (main path):
1. `main.py` -> load cfg/topology/data -> build strategy/server
2. Flower simulation starts
3. Strategy `initialize_parameters`
4. Repeat rounds:
   - `configure_fit`
   - client `fit`
   - `aggregate_fit`
   - `configure_evaluate`
   - client `evaluate`
   - `aggregate_evaluate`
   - strategy `evaluate` (centralized callback)
5. Final round -> `save_results`
6. Entrypoint writes raw output files

Backdoor run:
- Same orchestration, but client behavior replaced by `client_backdoor.py` (poisoning, ASR, boosting), and `run_summary.json` is added.

---

## 15. Quick troubleshooting checklist

1. No training progress seen:
- Verify topology includes self-node in each `pX`.
- Verify `local_train_cid` routing and head selection.

2. Runtime errors on CPU Windows with GPU checks:
- Confirm CPU fallback path in `main.py`/`main_backdoor.py` is active.

3. Strange aggregation behavior:
- Inspect whether many sampled clients have zero/near-zero effective weight.
- Check `early_local_train` setting.

4. Very strong initial accuracy:
- Pretraining may be loading from an existing checkpoint.
- Adjust `mix_alpha` and `noise_std`.

5. Backdoor attack seems inactive:
- Confirm malicious client id matches active participants in topology.
- Confirm poison rate and trigger logic are being applied.

---

## 16. Suggested hardening improvements

1. Replace `len(trainloader)` by true local sample counts for aggregation weighting.
2. Move malicious client IDs and attack hyperparameters to config YAML.
3. Save per-pool checkpoint from `pool_parameters[cli_ID]` instead of `selected_pool` inside loop.
4. Add explicit topology validation (self-inclusion, cid range checks, empty neighborhoods).
5. Add deterministic seeding in all random attack paths for strict reproducibility.

---

This document is intentionally code-grounded and aligned with the current implementation behavior in the listed files.