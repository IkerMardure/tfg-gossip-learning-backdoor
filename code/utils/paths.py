from pathlib import Path
import sys


def find_project_root(max_levels: int = 8) -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(max_levels):
        if (p / '.git').exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    # Fallback: assume two levels up from code/ is project root
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = find_project_root()


def data_root() -> Path:
    return PROJECT_ROOT / 'data' / 'datasets'


def results_root() -> Path:
    return PROJECT_ROOT / 'results' / 'outputs'


def paper_root() -> Path:
    return PROJECT_ROOT / 'paper'


def resolve_data_path(path_like: str | None) -> Path:
    if not path_like:
        return data_root()
    p = Path(path_like)
    if p.is_absolute():
        return p
    # If given relative, resolve against project root
    return PROJECT_ROOT / path_like


def resolve_results_path(path_like: str | None) -> Path:
    if not path_like:
        return results_root()
    p = Path(path_like)
    if p.is_absolute():
        return p
    normalized = str(path_like).replace('\\', '/').lstrip('./')
    if normalized.startswith('results/outputs/'):
        return PROJECT_ROOT / normalized
    if normalized.startswith('outputs/'):
        normalized = normalized[len('outputs/'):]
    return results_root() / normalized
