import matplotlib.pyplot as plt
import re
import ast

# Ruta del fichero
file_path = r"C:\Users\Iker\unibertsitatea\praktikak\GLow_poisoning\GLow-master\outputs\2025-06-17\18-17-52\raw.out"

# Leer y parsear el archivo
with open(file_path, 'r') as f:
    content = f.read()

# Utilidad para extraer listas desde el texto
def extract_list(name):
    pattern = r"\*\*{}:\s*(\([^)]+\)(?:\s+\([^)]+\))*)".format(re.escape(name))
    match = re.search(pattern, content)
    if not match:
        return []
    data_str = "[" + match.group(1).replace(") (", "), (") + "]"
    return ast.literal_eval(data_str)

# Extraer métricas
losses_distributed = extract_list("losses_distributed")
losses_centralized = extract_list("losses_centralized")
acc_distr_raw = extract_list("acc_distr")
metrics_centralized = extract_list("metrics_centralized")

# Procesar valores
rounds_loss_dist, loss_dist = zip(*losses_distributed)
rounds_loss_cent, loss_cent = zip(*losses_centralized)
rounds_acc_cent, acc_cent = zip(*metrics_centralized)
rounds_acc_dist, acc_dist = zip(*acc_distr_raw)

# Promediar precisión distribuida por ronda
avg_acc_dist = [sum(acc_list)/len(acc_list) for acc_list in acc_dist]

# --- Gráfico de pérdidas (loss) ---
plt.figure(figsize=(10, 5))
plt.plot(rounds_loss_dist, loss_dist, label="Distributed Loss", marker='o')
plt.plot(rounds_loss_cent, loss_cent, label="Centralized Loss", marker='x')
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Loss: Distributed vs Centralized")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_comparison.png")
plt.close()

# --- Gráfico de accuracy ---
plt.figure(figsize=(10, 5))
plt.plot(rounds_acc_dist, avg_acc_dist, label="Avg Distributed Accuracy", marker='o')
plt.plot(rounds_acc_cent, acc_cent, label="Centralized Accuracy", marker='x')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Accuracy: Distributed vs Centralized")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.close()

print("Gráficos guardados como 'loss_comparison.png' y 'accuracy_comparison.png'")
