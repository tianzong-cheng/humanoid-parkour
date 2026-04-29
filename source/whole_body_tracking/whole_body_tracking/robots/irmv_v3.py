"""IRMV Humanoid V3 robot configuration for motion tracking.

NOTE: This configuration uses TEMPORARY motor parameters based on Unitree G1.
TODO: Replace armature, stiffness, damping, effort limits, and velocity limits
with actual IRMV v3 motor specifications before production use.

Robot Structure:
- 22 actuated joints total
- Root link: pelvis
- Waist: waist_yaw_joint only (no roll/pitch)
- Arms: shoulder_pitch/roll/yaw + elbow (4 per arm, 8 total)
- Legs: hip_pitch/roll/yaw + knee + ankle_pitch/roll (6 per leg, 12 total)
- Waist: 1 joint

Note: Hip pitch joints have 30-degree tilted axes (design feature).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# ==============================================================================
# TEMPORARY Motor Parameters (Based on Unitree G1)
# TODO: Replace with actual IRMV v3 motor specifications
# ==============================================================================

# Armature values (kg·m²) - estimated from motor specs
# These values should be updated with actual motor rotor inertia
ARMATURE_5020 = 0.003609725  # Small motor (ankle, shoulder)
ARMATURE_7520_14 = 0.010177520  # Medium motor (hip pitch/yaw, waist)
ARMATURE_7520_22 = 0.025101925  # Large motor (hip roll, knee)

# PD gains calculated from natural frequency (10Hz) and damping ratio (2.0)
# These provide stable tracking for the estimated armature values
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ

# ==============================================================================
# Robot Articulation Configuration
# ==============================================================================

IRMV_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/irmv_description/urdf/irmv_v3.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Initial height estimated from leg kinematics
        # TODO: Adjust based on actual robot dimensions
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # Standing pose - bent knees for stability
            ".*_hip_pitch_joint": -0.25,
            ".*_knee_joint": 0.5,
            ".*_ankle_pitch_joint": -0.27,
            # Arms in natural position
            ".*_elbow_joint": 0.3,
            "left_shoulder_roll_joint": 0.15,
            "right_shoulder_roll_joint": -0.15,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Hip joints (pitch, roll, yaw)
        # Note: Hip pitch has 30° tilted axis - different dynamics
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            # TODO: Update with actual IRMV v3 motor specs
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        # Ankle joints
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        # Waist yaw (single joint, no roll/pitch like G1)
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        # Arm joints (shoulder pitch/roll/yaw, elbow)
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness=STIFFNESS_5020,
            damping=DAMPING_5020,
            armature=ARMATURE_5020,
        ),
    },
)

# ==============================================================================
# Action Scale Computation
# ==============================================================================

IRMV_V3_ACTION_SCALE = {}
for a in IRMV_V3_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            IRMV_V3_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

# ==============================================================================
# Motion Tracking Body Configuration
# ==============================================================================

IRMV_V3_ANCHOR_BODY_NAME = "pelvis"
"""The anchor body for motion tracking (pelvis is the root link)."""

IRMV_V3_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_shoulder_yaw_link",  # End effector for left arm
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_shoulder_yaw_link",  # End effector for right arm
]
"""List of body names for whole-body motion tracking.

Note: IRMV v3 uses shoulder_yaw_link as end effector (no wrist joints in current URDF).
This differs from G1 which has wrist_yaw_link as end effector.
"""

# ==============================================================================
# Joint Information
# ==============================================================================

IRMV_V3_NUM_JOINTS = 21
"""Total number of actuated joints."""

IRMV_V3_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]
"""Ordered list of joint names matching the controller configuration."""
