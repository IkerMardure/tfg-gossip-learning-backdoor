from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "img" / "colored_topologies"
NODE_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _is_isolated_topology(topology_path: Path) -> bool:
    return topology_path.stem.lower().startswith("isolated") or any(
        part.lower() == "isolated" for part in topology_path.parts
    )


def load_topology_graph(topology_path: Path) -> nx.Graph:
    with topology_path.open("r", encoding="utf-8") as file:
        topology_data = yaml.safe_load(file)

    pools = topology_data["pools"]
    graph = nx.Graph()

    for pool_name, neighbors in pools.items():
        node = int(pool_name[1:])
        graph.add_node(node)
        for neighbor in neighbors:
            neighbor_node = int(neighbor)
            graph.add_node(neighbor_node)
            if neighbor_node != node:
                graph.add_edge(node, neighbor_node)

    return graph


def draw_topology(topology_path: Path, target_node: int) -> Path:
    graph = load_topology_graph(topology_path)
    if target_node not in graph:
        raise ValueError(
            f"Node {target_node} is not present in topology {topology_path.name}"
        )

    nodes = sorted(graph.nodes)
    pos = nx.spring_layout(graph, seed=42)
    node_colors = [NODE_PALETTE[node % len(NODE_PALETTE)] for node in nodes]
    topology_name = topology_path.stem
    title = "Isolated" if _is_isolated_topology(topology_path) else f"Topology: {topology_name}"

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.7, edge_color="#8a8a8a")
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_color=node_colors,
        node_size=850,
        edgecolors="black",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node: str(node) for node in nodes},
        font_size=11,
        font_family="sans-serif",
        font_color="black",
        font_weight="bold",
    )

    plt.title(title, fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_name = "isolated_colored.png" if _is_isolated_topology(topology_path) else f"{topology_name}_node_{target_node}.png"
    output_path = OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
    return output_path


def parse_args():
    parser = ArgumentParser(
        description="Draw a topology from a YAML file with per-node colors."
    )
    parser.add_argument("topology", type=Path, help="Path to the topology YAML file")
    parser.add_argument("node", type=int, help="Node number used to validate the topology")
    return parser.parse_args()


def main():
    args = parse_args()
    topology_path = args.topology
    if not topology_path.is_file():
        raise FileNotFoundError(f"Topology file not found: {topology_path}")

    output_path = draw_topology(topology_path, args.node)
    print(f"Saved colored topology image to: {output_path}")


if __name__ == "__main__":
    main()