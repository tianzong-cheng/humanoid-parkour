"""IRMV V3 environment registrations."""

import gymnasium as gym

from . import agents, distillation_env_cfg, motion_tracking_env_cfg, perceptive_distillation_env_cfg

##
# Register Gym environments.
##

# ========================================
# Single Motion Whole-Body Tracking (Blind, Actor-Critic)
# ========================================

gym.register(
    id="IRMV_V3-MotionTracking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": motion_tracking_env_cfg.IRMV_V3_MotionTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRMV_V3_FlatPPORunnerCfg",
    },
)

gym.register(
    id="IRMV_V3-MotionTracking-WoStateEstimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": motion_tracking_env_cfg.IRMV_V3_MotionTrackingWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRMV_V3_FlatPPORunnerCfg",
    },
)

gym.register(
    id="IRMV_V3-MotionTracking-LowFreq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": motion_tracking_env_cfg.IRMV_V3_MotionTrackingLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRMV_V3_FlatLowFreqPPORunnerCfg",
    },
)

# ========================================
# Multi-Motion Distillation (Blind, Teacher-Student)
# ========================================

gym.register(
    id="IRMV_V3-Distillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": distillation_env_cfg.IRMV_V3_DistillationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:IRMV_V3_FlatDistillationRunnerCfg",
    },
)

gym.register(
    id="IRMV_V3-Distillation-WoStateEstimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": distillation_env_cfg.IRMV_V3_DistillationWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:IRMV_V3_FlatDistillationRunnerCfg",
    },
)

gym.register(
    id="IRMV_V3-Distillation-LowFreq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": distillation_env_cfg.IRMV_V3_DistillationLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:IRMV_V3_FlatDistillationRunnerCfg",
    },
)

# ========================================
# Multi-Motion Distillation with Perception (Teacher-Student + Depth Camera)
# ========================================

gym.register(
    id="IRMV_V3-PerceptiveDistillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": perceptive_distillation_env_cfg.IRMV_V3_PerceptiveDistillationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:IRMV_V3_FlatDistillationRunnerCfg",
    },
)

gym.register(
    id="IRMV_V3-PerceptiveDistillation-WoStateEstimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": perceptive_distillation_env_cfg.IRMV_V3_PerceptiveDistillationWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:IRMV_V3_FlatDistillationRunnerCfg",
    },
)

gym.register(
    id="IRMV_V3-PerceptiveDistillation-LowFreq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": perceptive_distillation_env_cfg.IRMV_V3_PerceptiveDistillationLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:IRMV_V3_FlatDistillationRunnerCfg",
    },
)

# ========================================
# Legacy environment IDs (for backward compatibility with G1 naming convention)
# ========================================

gym.register(
    id="Tracking-Flat-IRMV_V3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": motion_tracking_env_cfg.IRMV_V3_MotionTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRMV_V3_FlatPPORunnerCfg",
    },
)

gym.register(
    id="Distillation-Flat-IRMV_V3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": distillation_env_cfg.IRMV_V3_DistillationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:IRMV_V3_FlatDistillationRunnerCfg",
    },
)
