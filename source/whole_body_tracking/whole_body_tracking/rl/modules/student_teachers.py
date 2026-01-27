from __future__ import annotations

import os
import torch
from tensordict import TensorDict
from typing import Any

import onnxruntime as ort
from rsl_rl.modules import StudentTeacher

import wandb


class StudentTeachers(StudentTeacher):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        run_path_list: list[str] = [],
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            student_obs_normalization=student_obs_normalization,
            teacher_obs_normalization=teacher_obs_normalization,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        self.run_path_list = run_path_list
        self.teachers = []
        self.load_teachers(run_path_list)
        self.num_actions = num_actions

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        # Teacher policies are loaded in constructor, so we don't need to load a teacher policy here.
        if any("student" in key for key in state_dict):  # Load parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # Set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True  # Training resumes
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def load_teachers(self, run_path_list: list[str]):
        """Load exported ONNX models from Wandb runs specified in run_path_list.

        The ONNX policies are stored in "skillset" type output artifacts of the runs.
        """
        if not run_path_list:
            print("[INFO] No teacher run paths provided, skipping teacher loading.")
            return

        print(f"[INFO] Loading {len(run_path_list)} teacher policies from Wandb...")

        api = wandb.Api()
        self.teachers = []

        for i, run_path in enumerate(run_path_list):
            try:
                # Access the run
                run = api.run(run_path)

                # Find skillset artifacts
                artifacts = [a for a in run.logged_artifacts() if a.type == "skillset"]
                if not artifacts:
                    raise ValueError(f"No skillset artifacts found in run: {run_path}")

                artifact = artifacts[-1]  # Use the latest if there are multiple
                artifact_dir = artifact.download()

                # Find ONNX file in downloaded artifact directory
                onnx_files = [f for f in os.listdir(artifact_dir) if f.endswith(".onnx")]
                if not onnx_files:
                    raise ValueError(f"No ONNX files found in artifact: {artifact.name}")

                onnx_filename = onnx_files[0]
                onnx_path = os.path.join(artifact_dir, onnx_filename)
                teacher_session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])

                self.teachers.append(teacher_session)
                print(f"[INFO] Using artifact: {artifact.name} (version: {artifact.version})")

            except Exception as e:
                print(f"[ERROR] Failed to load teacher from {run_path}: {e}")
                raise

        print(f"[INFO] Successfully loaded {len(self.teachers)} teacher policies")

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        """Compute teacher actions using environment partitioning.

        With x teachers and N environments, first N/x envs use teacher 1,
        next N/x use teacher 2, etc. Handles remainder when N % x != 0.
        """
        with torch.no_grad():
            teacher_obs = self.get_teacher_obs(obs)
            teacher_obs = self.teacher_obs_normalizer(teacher_obs)

            num_envs = teacher_obs.shape[0]
            num_teachers = len(self.teachers)

            envs_per_teacher = num_envs // num_teachers
            remainder = num_envs % num_teachers

            actions = torch.zeros(num_envs, self.num_actions, device=teacher_obs.device)
            start_idx = 0
            for teacher_idx in range(num_teachers):
                end_idx = start_idx + envs_per_teacher + (1 if teacher_idx < remainder else 0)

                teacher = self.teachers[teacher_idx]

                partition_obs = teacher_obs[start_idx:end_idx]
                obs_numpy = partition_obs.cpu().numpy()
                teacher_output = teacher.run(None, {"obs": obs_numpy})[0]
                actions[start_idx:end_idx] = torch.from_numpy(teacher_output).to(teacher_obs.device)

                start_idx = end_idx

            return actions
