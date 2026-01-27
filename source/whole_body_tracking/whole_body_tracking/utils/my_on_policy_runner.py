from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", motion_registry: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.motion_registry = motion_registry
        self.collection_name = motion_registry.split("/")[-1].split(":")[0]
        self.last_artifact = None

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.policy,
                normalizer=self.alg.policy.actor_obs_normalizer,
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)

            if not wandb.run.settings._offline:
                if self.last_artifact is not None:
                    last_artifact = self.last_artifact
                    last_artifact.delete(delete_aliases=True)

                REGISTRY = "Parkour"
                COLLECTION = self.collection_name
                onnx_path = policy_path + filename
                logged_artifact = wandb.run.log_artifact(artifact_or_path=onnx_path, name=COLLECTION, type="skillset")
                wandb.run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
                self.last_artifact = logged_artifact
                print(f"[INFO]: Policy saved to wandb registry: {REGISTRY}/{COLLECTION}")

            # link the artifact registry to this run
            if self.motion_registry is not None:
                if not wandb.run.settings._offline:
                    wandb.run.use_artifact(self.motion_registry)
                self.motion_registry = None
