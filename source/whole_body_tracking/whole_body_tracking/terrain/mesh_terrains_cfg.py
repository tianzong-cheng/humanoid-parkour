from dataclasses import MISSING

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .mesh_terrains import obj_terrain


@configclass
class MeshObjTerrainCfg(SubTerrainBaseCfg):
    """Configuration for loading terrain from an OBJ file."""

    function = obj_terrain

    obj_path: str = MISSING
    """Path to the OBJ file relative to project root."""

    validate_bounds: bool = False
    """Whether to validate that the OBJ fits within terrain bounds. Defaults to True.

    If True, a warning will be issued if the OBJ extends beyond the terrain size.
    """
