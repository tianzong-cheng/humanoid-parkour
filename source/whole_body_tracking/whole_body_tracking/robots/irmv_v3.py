import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# TODO: These motor constants MUST be measured and replaced with actual IRMV V3 motor specs.
ARMATURE_6020 = 0.003609725
ARMATURE_8520 = 0.025101925

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_6020 = ARMATURE_6020 * NATURAL_FREQ**2
STIFFNESS_8520 = ARMATURE_8520 * NATURAL_FREQ**2

DAMPING_6020 = 2.0 * DAMPING_RATIO * ARMATURE_6020 * NATURAL_FREQ
DAMPING_8520 = 2.0 * DAMPING_RATIO * ARMATURE_8520 * NATURAL_FREQ

EFFORT_LIMIT_6020 = 20.0
EFFORT_LIMIT_8520 = 50.0

VELOCITY_LIMIT_6020 = 50.0
VELOCITY_LIMIT_8520 = 50.0

IRMV_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/irmv_v3/urdf/irmv_v3.urdf",
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
        pos=(0.0, 0.0, 0.76),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=EFFORT_LIMIT_8520,
            velocity_limit_sim=VELOCITY_LIMIT_8520,
            stiffness=STIFFNESS_8520,
            damping=DAMPING_8520,
            armature=ARMATURE_8520,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=EFFORT_LIMIT_6020,
            velocity_limit_sim=VELOCITY_LIMIT_6020,
            stiffness=2.0 * STIFFNESS_6020,
            damping=2.0 * DAMPING_6020,
            armature=2.0 * ARMATURE_6020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=EFFORT_LIMIT_8520,
            velocity_limit_sim=VELOCITY_LIMIT_8520,
            stiffness=STIFFNESS_8520,
            damping=DAMPING_8520,
            armature=ARMATURE_8520,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=EFFORT_LIMIT_6020,
            velocity_limit_sim=VELOCITY_LIMIT_6020,
            stiffness=STIFFNESS_6020,
            damping=DAMPING_6020,
            armature=ARMATURE_6020,
        ),
    },
)

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

##
# Motion tracking body configuration
##

IRMV_V3_ANCHOR_BODY_NAME = "torso_link"

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
    "right_shoulder_roll_link",
    "right_elbow_link",
]
