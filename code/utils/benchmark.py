"""
Benchmark utility for measuring GLow execution performance.
Tracks timing for key operations: data loading, rounds, and total execution.
"""

import time
import json
from pathlib import Path
from typing import Dict, Optional
from utils.logging import log_results


class ExecutionBenchmark:
    """Track and log execution timing for performance analysis."""
    
    def __init__(self):
        self.start_time = None
        self.timings: Dict[str, float] = {}
        self.round_times: list = []
        self.current_round_start = None
        
    def start(self):
        """Start overall execution timer."""
        self.start_time = time.time()
        
    def end(self) -> float:
        """End overall execution and return total time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def mark(self, label: str):
        """Mark a named checkpoint."""
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        self.timings[label] = elapsed
        
    def start_round(self):
        """Start timing a training round."""
        self.current_round_start = time.time()
        
    def end_round(self) -> float:
        """End round timing and return duration in seconds."""
        if self.current_round_start is None:
            return 0.0
        duration = time.time() - self.current_round_start
        self.round_times.append(duration)
        return duration
    
    def get_stats(self) -> Dict:
        """Get timing statistics."""
        if not self.round_times:
            return {
                "total_rounds": 0,
                "round_times": [],
                "min_round_time_sec": 0.0,
                "max_round_time_sec": 0.0,
                "avg_round_time_sec": 0.0,
            }
        
        return {
            "total_rounds": len(self.round_times),
            "round_times": [round(t, 2) for t in self.round_times],
            "min_round_time_sec": round(min(self.round_times), 2),
            "max_round_time_sec": round(max(self.round_times), 2),
            "avg_round_time_sec": round(sum(self.round_times) / len(self.round_times), 2),
        }
    
    def save_report(self, output_path: str):
        """Save benchmark report to JSON file."""
        total_time = self.end()
        checkpoint_times = self.timings.copy()
        
        report = {
            "total_execution_time_sec": round(total_time, 2),
            "checkpoints": {k: round(v, 2) for k, v in checkpoint_times.items()},
            "round_metrics": self.get_stats(),
        }
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
            
    def print_summary(self):
        """Print timing summary to console."""
        stats = self.get_stats()
        total_time = self.end()
        
        log_results("\n=== Execution Benchmark Summary ===", level="standard")
        log_results(f"Total execution time: {round(total_time, 2)}s", level="standard")
        if stats['total_rounds'] > 0:
            log_results(f"Total rounds completed: {stats['total_rounds']}", level="standard")
            log_results(f"Avg time per round: {stats['avg_round_time_sec']}s", level="standard")
            log_results(
                f"Min/Max round time: {stats['min_round_time_sec']}s / {stats['max_round_time_sec']}s",
                level="standard",
            )
        log_results("===================================\n", level="standard")


# Global benchmark instance
_benchmark = ExecutionBenchmark()


def get_benchmark() -> ExecutionBenchmark:
    """Get the global benchmark instance."""
    return _benchmark
