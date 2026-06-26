# 3D Loss Landscape Visualization Plan

You requested to implement a 3D surface plot (e.g., a paraboloid) that updates the coordinate trail iteratively during training to visualize the loss optimization. 

Because we are running on a GT 730 GPU, calculating the *true* 3D loss landscape of the full neural network during every step (which involves projecting the multi-million-dimensional weight space onto two random axes and running thousands of forward passes for the grid) would completely halt your 18.0 tokens/second training speed.

Instead, we can implement this in a lightweight, visually impressive way without slowing down the GPU.

## Proposed Implementation

### 1. The 3D Plotter (`loss_landscape_plotter.py`)
I will create a new standalone plotting script using `matplotlib` (which you already have installed). This script will:
- Generate a static 3D paraboloid surface (or a more complex non-convex mathematical surface) to conceptually represent the loss basin.
- Read your live `output/training_metrics_latest.csv` (or `.jsonl`) file that we implemented in the previous steps.
- Map the actual training progress (steps and true loss values) to X, Y coordinates on this synthetic 3D surface.
- Iteratively update the coordinate trail (the "ball rolling down the hill") as the real training loss decreases.

### 2. Integration with Training
In `train.py` and `auto_train.py`, we already have a hook that runs `training_log_plotter.py` asynchronously every 1,000 steps. 
I will add a second asynchronous subprocess call to trigger `loss_landscape_plotter.py` alongside the 4-panel chart.

This will output a `loss_landscape_latest.png` (or an interactive HTML file if you prefer `plotly`), updating smoothly as training progresses, without adding any overhead to the PyCUDA training loop.

## Open Questions

> [!QUESTION]
> **Plotting Library Preference:** Do you want me to build this using `matplotlib` (saving a `.png` file every 1000 steps) or `plotly` (saving an interactive `.html` file that you can rotate in your browser)? 
>
> **Conceptual vs True Space:** Do you approve of mapping the true loss onto a synthetic 3D mathematical surface (like a paraboloid) to keep training fast, as outlined above?
