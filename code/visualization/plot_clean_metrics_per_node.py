import ast
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# Marcadores para distinguir los clientes en el gráfico de Accuracy
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "8"]

def extract_list(content: str, name: str):
    """ Extrae las tuplas de datos del archivo .out usando regex """
    pattern = r"\*\*{}:\s*(\([^)]+\)(?:\s+\([^)]+\))*)".format(re.escape(name))
    match = re.search(pattern, content)
    if not match:
        return []
    # Convierte el formato (r, v) (r, v) en una lista de tuplas válida para Python
    data_str = "[" + match.group(1).replace(") (", "), (") + "]"
    return ast.literal_eval(data_str)

def main():
    if len(sys.argv) < 2:
        print("Uso: python plot_history_acc_loss.py <archivo_mnist_test_raw.out>")
        sys.exit(1)

    raw_path = Path(sys.argv[1])
    try:
        content = raw_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        sys.exit(1)

    # 1. EXTRACCIÓN DE DATOS [cite: 1143, 1147, 1149]
    acc_data = extract_list(content, "acc_distr")
    loss_data = extract_list(content, "losses_distributed")
    cid_data = extract_list(content, "cid")

    if not acc_data or not cid_data:
        print("Error: No se encontraron los campos necesarios en el archivo.")
        sys.exit(1)

    # 2. RECONSTRUCCIÓN DE SERIES POR NODO (Accuracy) [cite: 1147, 1149]
    series_acc = {}
    for (round_id, values), (_, cids) in zip(acc_data, cid_data):
        for val, cid in zip(values, cids):
            series_acc.setdefault(cid, []).append((round_id, val))

    # 3. CONFIGURACIÓN DEL GRÁFICO
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- SUBPLOT 1: ACCURACY (LOG SCALE) ---
    for index, cid in enumerate(sorted(series_acc.keys())):
        points = sorted(series_acc[cid])
        rounds, values = zip(*points)
        ax1.plot(
            rounds, 
            values, 
            label=f"Client {cid}", 
            marker=MARKERS[index % len(MARKERS)], 
            markersize=4,
            alpha=0.8
        )
    
    # Aplicar escala logarítmica y zoom (0.9 a 1.0) para notar la curva
    ax1.set_yscale("log")
    ax1.set_ylim(0.9, 1.0) 
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_yticks([0.9, 0.92, 0.94, 0.96, 0.98, 1.0])
    
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Evolution of Clean Accuracy per Node (Log Scale Zoom)")
    ax1.legend(loc="lower right", ncol=3, fontsize='small')
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    # --- SUBPLOT 2: LOSS (LINEAR SCALE) ---
    if loss_data:
        # En tu archivo, losses_distributed es un valor único por ronda [cite: 1143, 1144]
        loss_rounds, loss_values = zip(*sorted(loss_data))
        ax2.plot(loss_rounds, loss_values, color="#F58518", linewidth=2, label="Distributed Loss")
        
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Communication Round")
        ax2.set_title("Evolution of Distributed Loss")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    # 4. GUARDADO
    plt.tight_layout()
    output_name = raw_path.with_name("history_acc_loss_plot.png")
    plt.savefig(output_name, dpi=200)
    print(f"Gráfico generado exitosamente: {output_name}")

if __name__ == "__main__":
    main()