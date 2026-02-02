DEFAULT_CHECKPOINT = ""
DEFAULT_OUTPUT_DIR = ""
DEFAULT_FILENAME = ""

import argparse
import os
import sys
import torch
from tensordict import TensorDict

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Export policy to ONNX")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after Isaac Sim is initialized
from rsl_rl.modules import ActorCritic

from whole_body_tracking.utils.exporter import export_policy_as_onnx


def infer_architecture(state_dict):
    """Infer network architecture from checkpoint."""
    num_actor_obs = state_dict["actor.0.weight"].shape[1]
    num_critic_obs = state_dict["critic.0.weight"].shape[1]

    # Get actor hidden dims
    actor_hidden_dims = []
    layer_idx = 0
    while f"actor.{layer_idx}.weight" in state_dict:
        actor_hidden_dims.append(state_dict[f"actor.{layer_idx}.weight"].shape[0])
        layer_idx += 2
    num_actions = actor_hidden_dims[-1]
    actor_hidden_dims = actor_hidden_dims[:-1]

    # Get critic hidden dims
    critic_hidden_dims = []
    layer_idx = 0
    while f"critic.{layer_idx}.weight" in state_dict:
        critic_hidden_dims.append(state_dict[f"critic.{layer_idx}.weight"].shape[0])
        layer_idx += 2
    critic_hidden_dims = critic_hidden_dims[:-1]

    # Get noise std
    if "std" in state_dict:
        std_tensor = state_dict["std"]
        init_noise_std = std_tensor.mean().item() if std_tensor.numel() > 1 else std_tensor.item()
    else:
        init_noise_std = 1.0

    return {
        "num_actor_obs": num_actor_obs,
        "num_critic_obs": num_critic_obs,
        "num_actions": num_actions,
        "actor_hidden_dims": actor_hidden_dims,
        "critic_hidden_dims": critic_hidden_dims,
        "init_noise_std": init_noise_std,
    }


def load_policy(checkpoint_path):
    """Load ActorCritic from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    config = infer_architecture(state_dict)

    # Create observation space
    obs_dict = TensorDict(
        {
            "policy": torch.zeros(1, config["num_actor_obs"]),
            "critic": torch.zeros(1, config["num_critic_obs"]),
        },
        batch_size=[1],
    )
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    # Instantiate and load
    actor_critic = ActorCritic(
        obs=obs_dict,
        obs_groups=obs_groups,
        num_actions=config["num_actions"],
        actor_hidden_dims=config["actor_hidden_dims"],
        critic_hidden_dims=config["critic_hidden_dims"],
        init_noise_std=config["init_noise_std"],
    )
    actor_critic.load_state_dict(state_dict)
    normalizer = actor_critic.actor_obs_normalizer

    return actor_critic, normalizer


def main():
    try:
        actor_critic, normalizer = load_policy(args.checkpoint)

        export_policy_as_onnx(
            actor_critic=actor_critic,
            path=args.output_dir,
            normalizer=normalizer,
            filename=args.filename,
            verbose=False,
        )

        print(f"Exported: {os.path.join(args.output_dir, args.filename)}")

    except Exception as e:
        print(f"[Error] Failed exporting policy: {e}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)

    simulation_app.close()


if __name__ == "__main__":
    main()
