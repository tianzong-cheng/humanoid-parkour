from __future__ import annotations

import numpy as np
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def obj_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshObjTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate terrain by loading an OBJ file.

    The OBJ file is loaded as-is and positioned so that its origin (0,0,0)
    aligns with the center of the terrain cell. Following the pattern of other
    terrain generators, the mesh is positioned with its origin at the terrain
    cell center, and the system will apply appropriate translations for grid
    placement.

    Args:
        difficulty: The difficulty of the terrain (ignored for OBJ terrain).
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        FileNotFoundError: If the OBJ file cannot be found.
        ValueError: If the OBJ file cannot be loaded.
    """
    # Import here to avoid circular imports
    import os

    # Resolve the OBJ file path
    root_path = "."
    obj_full_path = os.path.join(root_path, cfg.obj_path)

    # Check if file exists
    if not os.path.exists(obj_full_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_full_path}")

    # Load the OBJ file
    try:
        mesh = trimesh.load_mesh(obj_full_path, process=False)
    except Exception as e:
        raise ValueError(f"Failed to load OBJ file {obj_full_path}: {e}")

    # Handle multiple meshes (concatenate them)
    if isinstance(mesh, trimesh.Scene):
        # Convert scene to single mesh
        mesh = trimesh.util.concatenate([geom for geom in mesh.geometry.values()])

    # Translate mesh so that OBJ origin (0,0,0) aligns with terrain cell center
    # This follows the pattern used by other terrain generators (e.g., flat_terrain)
    translation = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0])
    mesh.apply_translation(translation)

    # Validate bounds if requested (after translation)
    if cfg.validate_bounds and hasattr(mesh, "bounds"):
        bounds = mesh.bounds
        if bounds is not None:
            # Check if mesh stays within terrain cell bounds (0 to size)
            if bounds[0][0] < 0 or bounds[1][0] > cfg.size[0] or bounds[0][1] < 0 or bounds[1][1] > cfg.size[1]:
                import warnings

                warnings.warn(
                    f"OBJ mesh bounds [{bounds[0][0]:.2f}, {bounds[1][0]:.2f}] x "
                    f"[{bounds[0][1]:.2f}, {bounds[1][1]:.2f}] extend beyond terrain "
                    f"bounds [0, {cfg.size[0]:.2f}] x [0, {cfg.size[1]:.2f}]. "
                    "Consider adjusting terrain size or OBJ scale."
                )

    # Origin is at the center of the terrain cell (following flat_terrain pattern)
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])

    return [mesh], origin
