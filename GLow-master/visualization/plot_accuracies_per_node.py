import re
import os
import matplotlib.pyplot as plt

file = r"C:\Users\Iker\unibertsitatea\praktikak\GLow_TFG\GLow-master\outputs\2026-03-16 - 16_06 - Analysis - FC_5 - 10rounds - 5classes"
file_path = r"C:\Users\Iker\unibertsitatea\praktikak\GLow_TFG\GLow-master\outputs\2026-03-16 - 16_06 - Analysis - FC_5 - 10rounds - 5classes\mnist_ring_test_raw.out"
num_nodes = 5
start_node = 0  # Empieza por el nodo 0

metrics_centralized_pattern = re.compile(r"\*\*metrics_centralized:((?: \(\d+, [\d\.]+\))+)")

def parse_metric(tuples_string):
    return [(int(m[0]), float(m[1])) for m in re.findall(r"\((\d+), ([\d\.]+)\)", tuples_string)]

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

metrics_matches = parse_metric(metrics_centralized_pattern.search(text).group(1))

# Asociar cada accuracy al nodo correspondiente según el orden cíclico empezando por el nodo 0
node_acc = {i: [] for i in range(num_nodes)}
for idx, (round_id, acc) in enumerate(metrics_matches):
    node_id = (start_node + idx) % num_nodes
    node_acc[node_id].append((round_id, acc))

# Recorta los valores para que todos los nodos tengan la misma longitud
min_len = min(len(accs) for accs in node_acc.values())
for node_id in node_acc:
    node_acc[node_id] = node_acc[node_id][:min_len]

# Asigna colores fijos a cada nodo (el 2 siempre rojo)
node_colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'red', 3: 'tab:purple', 4: 'tab:green'}

plt.figure(figsize=(10, 6))
for node_id in range(num_nodes):
    if node_acc[node_id]:
        rounds, accs = zip(*node_acc[node_id])
        color = node_colors.get(node_id, f"C{node_id}")
        plt.plot(rounds, accs, marker='o', label=f"Client {node_id}", color=color)
plt.xlabel("Round")
plt.ylabel("Accuracy (trained node test)")
plt.title("Accuracy per node in Gossip Learning")
plt.legend()
plt.grid(True)
plt.tight_layout()
output_path = os.path.join(os.path.dirname(file_path), "accuracies_per_node.png")
plt.savefig(output_path, dpi=150)
plt.show()
