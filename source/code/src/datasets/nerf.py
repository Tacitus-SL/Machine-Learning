"""Posed images of a 3D scene and the camera rays through their pixels."""
import numpy as np
import torch
from torch.utils.data import Dataset

HELDOUT_VIEWS = range(100, 106)


def load_scene(path):
    """Load the 50 x 50 views of a scene: 100 for training, 6 held out.

    ``mic_50.npz`` holds the "mic" scene of the NeRF synthetic dataset
    (Mildenhall et al., 2020; model "mic" by up3d.de, CC-0): the 100
    training views and 6 test views, composited over black and
    downsampled from 800 x 800 to 50 x 50.

    Parameters
    ----------
    path : str or Path
        The ``.npz`` file with ``images`` (uint8, ``[106, 50, 50, 3]``),
        ``poses`` (``[106, 4, 4]`` camera-to-world matrices) and ``focal``.

    Returns
    -------
    dict
        ``'train_images'`` ``[100, 50, 50, 3]`` and ``'heldout_images'``
        ``[6, 50, 50, 3]``, float32 in [0, 1]; ``'train_poses'``,
        ``'heldout_poses'``; ``'focal'`` in pixels.
    """
    with np.load(path) as data:
        images = torch.from_numpy(data['images']).float() / 255.0
        poses = torch.from_numpy(data['poses']).float()
        focal = float(data['focal'])
    heldout = list(HELDOUT_VIEWS)
    train = [view for view in range(len(images)) if view not in heldout]
    return {'train_images': images[train], 'train_poses': poses[train],
            'heldout_images': images[heldout], 'heldout_poses': poses[heldout],
            'focal': focal}


class NeRFRayDataset(Dataset):
    """One camera ray per image pixel, with the pixel's color.

    Items are dicts with ``ray_origin`` and ``ray_direction`` (``[3]``,
    world space, unit direction) and ``target_rgb`` (``[3]``). The index may
    also be a list or tensor of indices; the item then holds a whole batch.
    """

    def __init__(self, images, poses, focal):
        images = torch.as_tensor(images, dtype=torch.float32)
        poses = torch.as_tensor(poses, dtype=torch.float32)
        n_views, self.height, self.width = images.shape[:3]
        self.colors = images.reshape(-1, 3)
        rows, columns = torch.meshgrid(torch.arange(self.height, dtype=torch.float32),
                                       torch.arange(self.width, dtype=torch.float32),
                                       indexing='ij')
        origins = torch.zeros(n_views, self.height, self.width, 3)
        ## Compute the ray of every pixel of every view:
        ## - camera-space direction (column - width / 2, -(row - height / 2), -focal) / focal;
        ## - rotate it by the pose (directions = R @ camera_direction), make it unit length;
        ## - every ray of a view starts at the camera position, the pose's translation.
        # -- start --
        d_cam = torch.stack([
            (columns - self.width / 2) / focal,
            -(rows - self.height / 2) / focal,
            torch.full_like(rows, -1.0),
        ], dim=-1)
        directions = d_cam.unsqueeze(0).expand(n_views, -1, -1, -1).clone()
        for view in range(n_views):
            R = poses[view, :3, :3]
            t = poses[view, :3, 3]
            d = directions[view].reshape(-1, 3)
            d = (R @ d.T).T
            d = d / d.norm(dim=-1, keepdim=True)
            directions[view] = d.reshape(self.height, self.width, 3)
            origins[view] = t
        self.origins = origins.reshape(-1, 3)
        self.directions = directions.reshape(-1, 3)

    def __len__(self):
        return len(self.colors)

    def __getitem__(self, index):
        return {'ray_origin': self.origins[index],
                'ray_direction': self.directions[index],
                'target_rgb': self.colors[index]}
