from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import yaml


def _load_topology_from_yaml(yaml_path: Path) -> Tuple[int, Dict[str, List[int]], List[int]]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Topology YAML not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    if not isinstance(cfg, dict):
        raise ValueError("Invalid topology YAML: expected a mapping at top level")

    if "num_clients" not in cfg or "pools" not in cfg:
        raise ValueError("Invalid topology YAML: expected 'num_clients' and 'pools'")

    num_clients = int(cfg["num_clients"])
    pools = cfg["pools"]
    if not isinstance(pools, dict):
        raise ValueError("Invalid topology YAML: 'pools' must be a mapping")

    clients_with_no_data = [int(cid) for cid in cfg.get("clients_with_no_data", [])]
    return num_clients, pools, clients_with_no_data


def _extract_edges(num_clients: int, pools: Dict[str, List[int]]) -> Iterable[Tuple[int, int]]:
    edges = set()

    for cli_id in range(num_clients):
        pool_key = f"p{cli_id}"
        if pool_key not in pools:
            raise ValueError(f"Invalid topology YAML: missing pool '{pool_key}'")

        neighbors = pools[pool_key]
        if not isinstance(neighbors, list) or len(neighbors) == 0:
            raise ValueError(f"Invalid topology YAML: pool '{pool_key}' must be a non-empty list")

        for raw_neighbor in neighbors[1:]:
            neighbor = int(raw_neighbor)
            if neighbor < 0 or neighbor >= num_clients:
                raise ValueError(
                    f"Invalid topology YAML: neighbor {neighbor} in '{pool_key}' out of range"
                )
            if neighbor == cli_id:
                continue
            edges.add(tuple(sorted((cli_id, neighbor))))

    return sorted(edges)


def _resolve_output_file(yaml_path: Path, output_path: str, image_format: str) -> Path:
    if output_path is None:
        return yaml_path.with_name(f"{yaml_path.stem}_graph.{image_format}")

    output = Path(output_path)
    if output.suffix:
        return output

    return output / f"{yaml_path.stem}_graph.{image_format}"


def _compute_layout(graph: nx.Graph, layout: str, seed: int):
    layout_name = layout.lower().strip()
    if layout_name == "spring":
        return nx.spring_layout(graph, seed=seed)
    if layout_name == "circular":
        return nx.circular_layout(graph)
    if layout_name == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    if layout_name == "shell":
        return nx.shell_layout(graph)
    raise ValueError("Unsupported layout. Use: spring, circular, kamada_kawai, shell")


def visualize_topology_from_yaml(
    yaml_path: str,
    output_path: str = None,
    image_format: str = "png",
    layout: str = "spring",
    seed: int = 42,
    dpi: int = 250,
    show_labels: bool = True,
) -> Path:
    yaml_file = Path(yaml_path)
    image_format = image_format.lower().strip()
    output_file = _resolve_output_file(yaml_file, output_path, image_format)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    num_clients, pools, clients_with_no_data = _load_topology_from_yaml(yaml_file)

    graph = nx.Graph()
    graph.add_nodes_from(range(num_clients))
    graph.add_edges_from(_extract_edges(num_clients, pools))

    isolated_nodes = [node for node, degree in dict(graph.degree()).items() if degree == 0]
    isolated_nodes = sorted(set(isolated_nodes).union(set(clients_with_no_data)))
    connected_nodes = [node for node in graph.nodes() if node not in set(isolated_nodes)]

    pos = _compute_layout(graph, layout, seed)
    plt.figure(figsize=(10, 8))

    nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.7, edge_color="#546E7A")

    if connected_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=connected_nodes,
            node_size=460,
            node_color="#2A9D8F",
            edgecolors="white",
            linewidths=1.2,
        )

    if isolated_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=isolated_nodes,
            node_size=500,
            node_color="#E76F51",
            edgecolors="white",
            linewidths=1.2,
        )

    if show_labels:
        nx.draw_networkx_labels(graph, pos, font_size=9, font_color="#1B1F23")

    plt.title(f"Topology Graph: {yaml_file.stem}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, format=output_file.suffix.lstrip("."), dpi=dpi, bbox_inches="tight")
    plt.close()

    return output_file