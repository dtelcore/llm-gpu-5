import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _safe_float(v):
    try:
        return float(v)
    except:
        return None

def read_metrics(log_dir="output"):
    p = Path(log_dir)
    paths = list(p.glob("*.jsonl")) + list(p.glob("*.csv"))
    if not paths:
        return []
    
    # Pick the newest
    latest_path = sorted(paths, key=lambda x: x.stat().st_mtime)[-1]
    
    losses = []
    if latest_path.suffix == ".jsonl":
        with latest_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "loss" in data and data["loss"] is not None:
                        losses.append(float(data["loss"]))
                except Exception:
                    pass
    elif latest_path.suffix == ".csv":
        with latest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    val = _safe_float(row.get("loss"))
                    if val is not None:
                        losses.append(val)
                except Exception:
                    pass
    return losses

def main():
    losses = read_metrics()
    if not losses:
        print("[INFO] No loss metrics found yet.")
        return

    # Create synthetic grid
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    # Simple paraboloid: Z = x^2 + 1.5 y^2
    Z = X**2 + 1.5 * Y**2
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, linewidth=0, antialiased=True)
    
    loss_arr = np.array(losses)
    min_loss, max_loss = np.min(loss_arr), np.max(loss_arr)
    
    # Map loss history to a path
    if max_loss > min_loss:
        normalized_loss = (loss_arr - min_loss) / (max_loss - min_loss)
    else:
        normalized_loss = np.zeros_like(loss_arr)
        
    # Start at radius 4, go to 0 as loss decreases
    radii = 4.0 * np.sqrt(normalized_loss)
    # Add a spiral effect to simulate gradient descent
    angles = np.linspace(0, 4 * np.pi, len(losses)) 
    
    path_X = radii * np.cos(angles)
    path_Y = radii * np.sin(angles) / np.sqrt(1.5)  # Scale to match the elliptical bowl Z
    path_Z = path_X**2 + 1.5 * path_Y**2
    
    # Plot the trail
    step_size = max(1, len(losses) // 50)
    ax.plot(path_X, path_Y, path_Z, color='red', linewidth=2, label='Training Trajectory')
    ax.scatter(path_X[::step_size], path_Y[::step_size], path_Z[::step_size], color='black', s=10)
    
    # Highlight current state
    ax.scatter([path_X[-1]], [path_Y[-1]], [path_Z[-1]], color='red', s=100, label=f'Current Loss: {losses[-1]:.4f}')
    
    ax.set_title("3D Loss Landscape Optimization Trajectory")
    ax.set_xlabel("PCA Axis 1 (Conceptual)")
    ax.set_ylabel("PCA Axis 2 (Conceptual)")
    ax.set_zlabel("Loss Basin (Conceptual)")
    ax.legend()
    
    save_path = "output/loss_landscape_latest.png"
    os.makedirs("output", exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"[INFO] 3D loss landscape saved to {save_path}")

if __name__ == "__main__":
    main()
