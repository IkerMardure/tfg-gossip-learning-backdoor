import sys
from pathlib import Path

# Force the working directory where you ran the command into Python's path search
if "" not in sys.path:
    sys.path.insert(0, "")

import ast
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Now import your utility module safely
try:
    from utils.logging import log_results
except ModuleNotFoundError:
    # Fallback if the path environment is locked down
    def log_results(msg, level="minimal"):
        print(f"[{level.upper()}] {msg}")

# --- THESIS PLOT CONFIGURATION (Larger Text) ---
plt.rcParams.update({
    'font.size': 20,          # Base size
    'axes.titlesize': 20,     # Plot titles
    'axes.labelsize': 20,     # X and Y axis labels
    'xtick.labelsize': 18,    # X axis tick labels
    'ytick.labelsize': 18,    # Y axis tick labels
    'legend.fontsize': 18     # Legend text
})

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
        raise SystemExit("Uso: python plot_history_acc_loss.py <archivo_mnist_test_raw.out>")

    raw_path = Path(sys.argv[1])
    try:
        content = raw_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        sys.exit(1)

    print(f"--- Processing: {raw_path.name} ---")

    # 1. EXTRACCIÓN DE DATOS
    acc_data = extract_list(content, "acc_distr")
    loss_data = extract_list(content, "losses_distributed")
    cid_data = extract_list(content, "cid")

    if not acc_data or not cid_data:
        print("Error: No se encontraron los campos necesarios en el archivo.")
        sys.exit(1)

    # 2. RECONSTRUCCIÓN DE SERIES POR NODO (Accuracy)
    series_acc = {}
    for (round_id, values), (_, cids) in zip(acc_data, cid_data):
        for val, cid in zip(values, cids):
            series_acc.setdefault(cid, []).append((round_id, val))

    # 3. CONFIGURACIÓN DEL GRÁFICO (Side-by-Side Layout)
    fig, axes = plt.subplots(1, 2, figsize=(22, 6))

    # --- SUBPLOT 1: ACCURACY ---
    ax1 = axes[0]
    for index, cid in enumerate(sorted(series_acc.keys())):
        points = sorted(series_acc[cid])
        rounds, values = zip(*points)
        point_marker = MARKERS[index % len(MARKERS)]
        ax1.plot(
            rounds, 
            values, 
            label=f"Client {cid}", 
            marker=point_marker, 
            linewidth=1.5,
            markeredgewidth=1.0,
            alpha=0.8
        )
    
    # Usar escala lineal y ajustar rango para observar la evolución
    ax1.set_ylim(0.85, 1.00)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Round")
    ax1.set_title("Evolution of Clean Accuracy per Node")
    
    # Place legend outside Plot 1 (between the two plots)
    ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0.)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- SUBPLOT 2: LOSS ---
    ax2 = axes[1]
    if loss_data:
        loss_rounds, loss_values = zip(*sorted(loss_data))
        ax2.plot(loss_rounds, loss_values, color="#F58518", linewidth=2.5, label="Distributed Loss")
        
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Round")
        ax2.set_title("Evolution of Distributed Loss")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle='--', alpha=0.7)

    # 4. GUARDADO
    # Use constrained layout engine to prevent outside legend clipping
    fig.set_layout_engine('constrained')
    output_name = raw_path.with_name("history_acc_loss_plot.png")
    
    output_name.parent.mkdir(parents=True, exist_ok=True)
    # Save with high resolution (300 DPI) for printing/thesis text crispness
    fig.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"SUCCESS: Saved plot to {output_name}")
    try:
        log_results(f"Saved accuracy/loss metrics plot to {output_name}", level="minimal")
    except Exception:
        pass

if __name__ == "__main__":
    main()