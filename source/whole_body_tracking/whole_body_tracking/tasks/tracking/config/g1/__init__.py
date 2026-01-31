import gymnasium as gym

from . import agents, distillation_env_cfg, motion_tracking_env_cfg, perceptive_distillation_env_cfg

##
# Register Gym environments.
##

# ========================================
# Single Motion Whole-Body Tracking (Blind, Actor-Critic)
# ========================================

gym.register(
    id="G1-MotionTracking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": motion_tracking_env_cfg.G1MotionTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="G1-MotionTracking-WoStateEstimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": motion_tracking_env_cfg.G1MotionTrackingWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="G1-MotionTracking-LowFreq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": motion_tracking_env_cfg.G1MotionTrackingLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg",
    },
)

# ========================================
# Multi-Motion Distillation (Blind, Teacher-Student)
# ========================================

gym.register(
    id="G1-Distillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": distillation_env_cfg.G1DistillationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg",
    },
)

gym.register(
    id="G1-Distillation-WoStateEstimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": distillation_env_cfg.G1DistillationWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg",
    },
)

gym.register(
    id="G1-Distillation-LowFreq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": distillation_env_cfg.G1DistillationLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg",
    },
)

# ========================================
# Multi-Motion Distillation with Perception (Teacher-Student + Depth Camera)
# ========================================

gym.register(
    id="G1-PerceptiveDistillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": perceptive_distillation_env_cfg.G1PerceptiveDistillationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg",
    },
)

gym.register(
    id="G1-PerceptiveDistillation-WoStateEstimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": perceptive_distillation_env_cfg.G1PerceptiveDistillationWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg",
    },
)

gym.register(
    id="G1-PerceptiveDistillation-LowFreq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": perceptive_distillation_env_cfg.G1PerceptiveDistillationLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg",
    },
)

# ========================================
# Legacy environment IDs (for backward compatibility)
# These will be deprecated in future versions
# ========================================

gym.register(
    id="Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": motion_tracking_env_cfg.G1MotionTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Distillation-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": distillation_env_cfg.G1DistillationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg",
    },
)
