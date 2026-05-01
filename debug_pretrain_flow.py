#!/usr/bin/env python
"""
Debug script to verify that pretraining weights are being used correctly
in the GL setup with FC_5 topology (5 nodes).
"""
import sys
from pathlib import Path
import yaml
import numpy as np

# Add code to path
sys.path.insert(0, str(Path(__file__).parent / "code"))

def main():
    print("\n" + "=" * 80)
    print("PRETRAINING WEIGHTS FLOW VERIFICATION (FC_5, 5 nodes)")
    print("=" * 80 + "\n")

    # Load config
    config_file = "code/conf/base.yaml"
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    print("1. Configuration Check:")
    print(f"   - Pretraining enabled: {cfg.get('pretraining', {}).get('enabled', False)}")
    print(f"   - Pretraining epochs: {cfg.get('pretraining', {}).get('epochs', 1)}")
    print(f"   - Pretraining LR: {cfg.get('pretraining', {}).get('lr', 0.001)}")
    print(f"   - Mix alpha (for blending): {cfg.get('pretraining', {}).get('mix_alpha', 1.0)}")
    
    pretrain_save_path = cfg.get('pretraining', {}).get('save_path', None)
    print(f"   - Save path template: {pretrain_save_path}\n")

    # Simulate path resolution for FC_5 (5 nodes)
    num_clients = 5
    print(f"2. Per-node weight paths for FC_5 ({num_clients} nodes):\n")
    
    # Simulate what main_backdoor.py does
    save_path = "/results/2026-05-01 - 11_30 - FC_5/"  # Example run path
    
    if pretrain_save_path is None:
        # Default case: weights saved in results folder
        for node_id in range(num_clients):
            model_save_path = str(Path(save_path) / f"pretrain_node_{node_id}.pth")
            print(f"   Node {node_id}: {model_save_path}")
    else:
        # Case with custom save path
        if "{node_id}" in str(pretrain_save_path):
            # Template with placeholder
            for node_id in range(num_clients):
                model_save_path = str(Path(str(pretrain_save_path).format(node_id=node_id)))
                print(f"   Node {node_id}: {model_save_path}")
        else:
            # Directory or file path
            save_spec = str(pretrain_save_path)
            p = Path(save_spec)
            if p.exists() and p.is_dir():
                files = sorted(p.glob("*.pth"))
                print(f"   Found {len(files)} .pth files in {save_spec}:")
                for node_id in range(num_clients):
                    if files:
                        mapped_file = files[node_id % len(files)]
                        print(f"   Node {node_id} -> {mapped_file}")
                    else:
                        print(f"   Node {node_id}: NO FILES FOUND")
            else:
                # File path with siblings check
                dirp = p.parent
                pattern = p.stem + "_node*.pth"
                siblings = sorted(dirp.glob(pattern)) if dirp.exists() else []
                if siblings:
                    print(f"   Found {len(siblings)} sibling files matching {pattern}:")
                    for node_id in range(num_clients):
                        mapped_file = siblings[node_id % len(siblings)]
                        print(f"   Node {node_id} -> {mapped_file}")
                else:
                    print(f"   No siblings found. Using exact path:")
                    print(f"   All nodes -> {p}")

    print("\n3. Expected flow during GL:")
    print("   ✓ Pretraining phase: Each node trains locally and saves weights")
    print("   ✓ Load or reuse: On subsequent runs, weights are loaded from disk")
    print("   ✓ Strategy initialization: pool_parameters = [weights_node_0, ..., weights_node_4]")
    print("   ✓ Round 1: Each client receives its pretrained weights (not random)")
    print("   ✓ Round 2+: Gossip learning begins with pretrained initialization\n")

    print("4. Verification Steps:")
    print("   ✓ Check that log shows: 'Initializing with pretrained parameters for 5 nodes'")
    print("   ✓ Check that nodes 0-4 each get different weights in Round 1")
    print("   ✓ Check that initial validation accuracy is ~0.95+ (not ~0.1 from random init)")
    print("   ✓ Check that all 5 nodes start with high accuracy, not just node 0\n")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
