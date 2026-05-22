from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "img" / "colored_topologies"

# Matplotlib's default tab10 colors (matches the line plots used in the repo)
TAB10_COLORS = plt.cm.tab10.colors[:8]


def save_colored_topology(graph, filename, pos, title):
    plt.figure(figsize=(4, 4))

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=TAB10_COLORS,
        node_size=800,
        edgecolors="black",
    )
    nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.6, edge_color="gray")
    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=12,
        font_family="sans-serif",
        font_color="white",
        font_weight="bold",
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / filename, dpi=300, transparent=True)
    plt.close()


def build_topologies():
    # 1. Fully connected
    graph_fc = nx.complete_graph(8)
    save_colored_topology(
        graph_fc,
        "fc_colored.png",
        nx.circular_layout(graph_fc),
        "Fully Connected",
    )

    # 2. Ring
    graph_ring = nx.cycle_graph(8)
    save_colored_topology(
        graph_ring,
        "ring_colored.png",
        nx.circular_layout(graph_ring),
        "Ring Topology",
    )

    # 3. Star (Client 0 is center)
    graph_star = nx.star_graph(7)
    save_colored_topology(
        graph_star,
        "star_colored.png",
        nx.spring_layout(graph_star, seed=42),
        "Star Topology",
    )

    # 4. SBM
    graph_sbm = nx.Graph()
    graph_sbm.add_nodes_from(range(8))

    # Community A (0-3) fully connected
    for i in range(4):
        for j in range(i + 1, 4):
            graph_sbm.add_edge(i, j)

    # Community B (4-7) fully connected
    for i in range(4, 8):
        for j in range(i + 1, 8):
            graph_sbm.add_edge(i, j)

    # The Gateway Link
    graph_sbm.add_edge(3, 4)

    pos_sbm = {
        0: [-1, 1],
        1: [-2, 0],
        2: [-1, -1],
        3: [-0.3, 0],
        4: [0.3, 0],
        5: [1, 1],
        6: [2, 0],
        7: [1, -1],
    }
    save_colored_topology(
        graph_sbm,
        "sbm_colored.png",
        pos_sbm,
        "Stochastic Block Model",
    )


if __name__ == "__main__":
    build_topologies()
    print(f"Saved colored topology images to: {OUTPUT_DIR}")