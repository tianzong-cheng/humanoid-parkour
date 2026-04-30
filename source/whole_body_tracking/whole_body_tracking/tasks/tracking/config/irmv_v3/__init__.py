import gymnasium as gym

from . import agents, motion_tracking_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-IRMV-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": motion_tracking_env_cfg.IRMVMotionTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRMVFlatPPORunnerCfg",
    },
)
