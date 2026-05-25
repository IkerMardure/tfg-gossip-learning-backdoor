#!/usr/bin/env python3
import re
import ast
import json
import os
import argparse
from statistics import mean


def extract_section(content, name):
    m = re.search(r"\*\*%s:\s*(.*?)(?=\n\*\*|\Z)" % re.escape(name), content, re.S)
    return m.group(1).strip() if m else ""


def parse_number_tuples(section):
    # matches patterns like (123, 0.456)
    pairs = re.findall(r"\((\d+),\s*([0-9eE+\-\.]+)\)", section)
    return [(int(r), float(v)) for r, v in pairs]


def parse_list_tuples(section):
    # matches patterns like (123, [..]) capturing the list text
    pairs = re.findall(r"\((\d+),\s*(\[[^\)]*\])\)", section, re.S)
    out = []
    for r, list_text in pairs:
        try:
            val = ast.literal_eval(list_text)
        except Exception:
            # fallback: try to replace Java-style true/false etc
            val = None
        out.append((int(r), val))
    return out


def parse_exec_time(content):
    m = re.search(r"\*\*Exec_time_secs:\s*([0-9eE+\-\.]+)", content)
    return float(m.group(1)) if m else None


def load_topology_pools(topology_path):
    if not topology_path or not os.path.exists(topology_path):
        return None
    try:
        import yaml
        with open(topology_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            pools = data.get('pools') if isinstance(data, dict) else None
            return pools
    except Exception:
        # simple fallback parser
        pools = {}
        with open(topology_path, 'r', encoding='utf-8') as f:
            cur = None
            for line in f:
                line = line.rstrip('\n')
                m = re.match(r"^(\s*)(p\d+):\s*$", line)
                if m:
                    cur = m.group(2)
                    pools[cur] = []
                    continue
                m2 = re.match(r"^\s*-\s*(\d+)", line)
                if m2 and cur is not None:
                    pools[cur].append(int(m2.group(1)))
        return pools


def find_topology_yaml_for_run(run_dir):
    # try to find a matching topology yaml under code/conf/topologies/analysis
    base = os.path.basename(run_dir)
    candidates = []
    search_root = os.path.join(os.getcwd(), 'code', 'conf', 'topologies', 'analysis')
    if not os.path.isdir(search_root):
        return None
    for root, _, files in os.walk(search_root):
        for fn in files:
            if not fn.lower().endswith('.yaml'):
                continue
            name = fn.lower()
            if name.replace('.yaml','') in base.lower() or base.lower().replace(' ','_').endswith(name.replace('.yaml','')):
                candidates.append(os.path.join(root, fn))
    return candidates[0] if candidates else None


def parse_file(path, topology_yaml=None):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics_section = extract_section(content, 'metrics_centralized')
    metrics = parse_number_tuples(metrics_section)
    final_round = max((r for r, _ in metrics), default=None)
    centralized_acc = None
    if metrics:
        centralized_acc = max(metrics, key=lambda x: x[0])[1]

    asr_section = extract_section(content, 'asr')
    asr_list = parse_list_tuples(asr_section)
    cid_section = extract_section(content, 'cid')
    cid_list = parse_list_tuples(cid_section)
    acc_distr_section = extract_section(content, 'acc_distr')
    acc_distr_list = parse_list_tuples(acc_distr_section)

    # pick final entries
    final_asr = None
    final_cid = None
    final_acc_distr = None
    for r, v in asr_list:
        if r == final_round:
            final_asr = v
    for r, v in cid_list:
        if r == final_round:
            final_cid = v
    for r, v in acc_distr_list:
        if r == final_round:
            final_acc_distr = v

    exec_time = parse_exec_time(content)

    # build mapping client_id -> asr
    asr_by_client = {}
    if final_asr is not None:
        if final_cid is not None and len(final_asr) == len(final_cid):
            for idx, client in enumerate(final_cid):
                try:
                    asr_by_client[int(client)] = float(final_asr[idx])
                except Exception:
                    pass
        else:
            # assume list indexed by client id
            for idx, val in enumerate(final_asr):
                try:
                    asr_by_client[int(idx)] = float(val)
                except Exception:
                    pass

    attacker_client = None
    attacker_asr = None
    if asr_by_client:
        attacker_client = max(asr_by_client.items(), key=lambda x: x[1])[0]
        attacker_asr = asr_by_client[attacker_client]

    # attacker clean acc from acc_distr
    attacker_clean_acc = None
    if final_acc_distr is not None:
        if isinstance(final_acc_distr, list):
            # if length equals num clients, index by client id
            if attacker_client is not None and attacker_client < len(final_acc_distr):
                attacker_clean_acc = float(final_acc_distr[attacker_client])
            elif final_cid is not None and len(final_acc_distr) == len(final_cid):
                # align by cid order
                idx = None
                if attacker_client is not None:
                    for i, c in enumerate(final_cid):
                        if int(c) == int(attacker_client):
                            idx = i
                            break
                if idx is not None:
                    attacker_clean_acc = float(final_acc_distr[idx])

    # load topology pools and compute local ASR using pool of attacker
    pools = None
    if topology_yaml:
        pools = load_topology_pools(topology_yaml)
    else:
        guessed = find_topology_yaml_for_run(os.path.dirname(path))
        if guessed:
            pools = load_topology_pools(guessed)

    local_asr = None
    if pools and attacker_client is not None:
        pool_key = f'p{attacker_client}'
        neigh = pools.get(pool_key) or pools.get(pool_key.lower())
        if neigh:
            vals = [asr_by_client.get(int(n)) for n in neigh if asr_by_client.get(int(n)) is not None]
            if vals:
                local_asr = mean(vals)

    # global ASR: mean over available asr_by_client values
    global_asr = mean(list(asr_by_client.values())) if asr_by_client else None

    return {
        'run_path': path,
        'final_round': final_round,
        'centralized_acc': centralized_acc,
        'attacker_client': attacker_client,
        'attacker_asr': attacker_asr,
        'attacker_clean_acc': attacker_clean_acc,
        'local_asr': local_asr,
        'global_asr': global_asr,
        'exec_time_secs': exec_time,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('paths', nargs='+', help='Path(s) to mnist_test_raw.out files or parent run directories')
    p.add_argument('--topology', help='Optional path to topology yaml to use for neighbourhoods')
    args = p.parse_args()

    results = []
    for path in args.paths:
        if os.path.isdir(path):
            candidate = os.path.join(path, 'mnist_test_raw.out')
            if not os.path.exists(candidate):
                # try to find any mnist_test_raw.out under this dir
                for root, _, files in os.walk(path):
                    for fn in files:
                        if fn == 'mnist_test_raw.out':
                            candidate = os.path.join(root, fn)
                            break
            path = candidate
        if not os.path.exists(path):
            print(json.dumps({'error': f'path not found: {path}'}))
            continue
        res = parse_file(path, topology_yaml=args.topology)
        results.append(res)

    if len(results) == 1:
        print(json.dumps(results[0], indent=2))
    else:
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
