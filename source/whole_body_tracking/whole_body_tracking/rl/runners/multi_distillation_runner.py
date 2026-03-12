from __future__ import annotations

import os

from rsl_rl.runners import DistillationRunner

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MultiDistillationRunner(DistillationRunner):
    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the model and export ONNX policy to Wandb registry."""
        super().save(path, infos)

        # TODO: Support multi-distillation
        return
        if self.logger.logger_type in ["wandb"]:
            # Export ONNX policy
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"

            export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.policy,
                normalizer=self.alg.policy.student_obs_normalizer,
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)

            if not wandb.run.settings._offline:
                # Delete previous artifact if exists
                if hasattr(self, "last_artifact") and self.last_artifact is not None:
                    self.last_artifact.delete(delete_aliases=True)

                # Upload to Wandb registry
                REGISTRY = "Parkour"
                COLLECTION = "universal_parkour"
                onnx_path = os.path.join(policy_path, filename)

                logged_artifact = wandb.run.log_artifact(artifact_or_path=onnx_path, name=COLLECTION, type="universal")
                wandb.run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
                self.last_artifact = logged_artifact
                print(f"[INFO]: Universal parkour policy saved to wandb registry: {REGISTRY}/{COLLECTION}")

            # TODO: Use motion artifacts
