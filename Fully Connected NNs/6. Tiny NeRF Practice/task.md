# Tiny NeRF: a 3D Scene from Photos

Cast camera rays, encode 3D positions, predict density and color with a small network, and render the rays by volume rendering. The notebook trains on 100 rendered photos of the 'mic' scene of the NeRF synthetic dataset (CC-0) at 50 x 50 pixels and renders 6 held-out views. Edit `source/code/src/datasets/nerf.py`, `source/code/src/models/feedforward/nerf.py`, and the marked cells in `task.ipynb`. The test run uses the same data and only trains a few steps.
