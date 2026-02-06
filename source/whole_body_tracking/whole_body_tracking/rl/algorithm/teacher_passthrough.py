"""TeacherPassThrough algorithm for validating teacher policies before distillation.

This algorithm provides a minimal validation-only mode that passes observations directly
to teacher policies and returns their outputs as actions, without any training. It is
designed to verify that teacher policies work correctly (proper loading, inference,
environment partitioning) before using them in distillation training.

Key features:
- Reuses StudentTeachers.evaluate() for teacher inference with environment partitioning
- Implements RSL-RL algorithm interface with no-op training
- Validates teacher policy loading and integration
- Records transitions for compatibility with RSL-RL runners
- Updates observation normalizers for consistency with training mode

Usage:
    This algorithm should be used with DistillationRunner and a StudentTeachers policy
    that has been initialized with valid teacher run paths. It will raise an error if
    teacher policies are not properly loaded.
"""

from __future__ import annotations

import torch

from rsl_rl.storage import RolloutStorage

from whole_body_tracking.rl.modules import StudentTeachers


class TeacherPassThrough:
    """Validation-only algorithm that passes observations to teacher policies.

    This algorithm implements the RSL-RL algorithm interface but performs no training.
    It is designed for validating teacher policies before distillation training by:
    1. Loading teacher policies from WandB run paths
    2. Partitioning environments across teachers
    3. Running teacher inference and returning actions
    4. Recording transitions and updating normalizers (but no gradient updates)

    The algorithm verifies that:
    - Teacher policies load successfully from ONNX artifacts
    - Environment partitioning works correctly
    - Inference produces valid actions
    - Episode rewards are reasonable

    Args:
        policy: StudentTeachers policy instance with loaded teacher policies
        storage: RolloutStorage for recording transitions
        device: Device to run computations on (cpu or cuda)
        multi_gpu_cfg: Multi-GPU configuration (unused, for interface compatibility)
        **kwargs: Additional arguments (ignored)

    Raises:
        ValueError: If teacher policies are not loaded in the policy module
    """

    def __init__(
        self,
        policy: StudentTeachers,
        storage: RolloutStorage,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        """Initialize TeacherPassThrough algorithm.

        Args:
            policy: StudentTeachers policy instance with loaded teachers
            storage: RolloutStorage for recording transitions
            device: Device to run computations on
            multi_gpu_cfg: Multi-GPU configuration (unused)
            **kwargs: Additional arguments (ignored)
        """
        # Verify that teacher policies are loaded
        if not policy.loaded_teacher:
            raise ValueError(
                "Teacher policies not loaded in policy module. "
                "Ensure run_path_list is provided in policy config and teachers were successfully loaded from WandB."
            )

        self.policy = policy
        self.storage = storage
        self.device = device
        self.multi_gpu_cfg = multi_gpu_cfg

        # Create transition object for recording state transitions
        self.transition = storage.Transition()

        # Track hidden states for recurrent policy compatibility
        self.last_hidden_states = None

        # Dummy learning rate for runner logging compatibility
        self.learning_rate = 0.0

        # Dummy optimizer for checkpoint saving compatibility
        # The runner tries to save optimizer state_dict, but no training occurs
        # Create a minimal optimizer with no parameters to optimize
        self.optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True, device=device)], lr=0.0)

        # Initialize policy distribution for logging compatibility
        # The runner expects policy.action_std to exist, which requires distribution to be set
        # Create dummy student observations and call _update_distribution to initialize it
        dummy_student_obs = torch.zeros(1, policy.student[0].in_features, device=device)
        policy._update_distribution(dummy_student_obs)

        # Log initialization info
        num_teachers = len(policy.teachers) if hasattr(policy, "teachers") else 0
        print(f"[INFO] TeacherPassThrough initialized with {num_teachers} teacher(s)")
        print("[INFO] This is a validation-only mode - no training will occur")

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Get actions from teacher policies using environment partitioning.

        This method calls policy.evaluate() which:
        1. Extracts teacher observations from the observation dict
        2. Normalizes observations using the policy's normalizer
        3. Partitions environments across available teachers
        4. Runs ONNX inference for each teacher on their assigned environments
        5. Combines results into a single action tensor

        Args:
            obs: Observation dict containing teacher and student observations
            critic_obs: Critic observations (unused, for interface compatibility)

        Returns:
            Actions tensor of shape (num_envs, num_actions)
        """
        with torch.no_grad():
            # Record hidden states if policy is recurrent
            if self.policy.is_recurrent:
                self.transition.hidden_states = self.policy.get_hidden_states()

            # Use StudentTeachers.evaluate() for teacher inference with partitioning
            actions = self.policy.evaluate(obs)

            # Record observations and actions in transition
            self.transition.observations = obs
            self.transition.actions = actions

            # For distillation storage, privileged_actions are the teacher actions
            if self.storage.training_type == "distillation":
                self.transition.privileged_actions = actions

        return actions

    def process_env_step(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> None:
        """Process environment step by updating normalizers and recording transitions.

        This method:
        1. Updates observation normalizers (for consistency with training mode)
        2. Records rewards and dones in transition
        3. Adds complete transition to storage
        4. Clears transition for next step
        5. Resets policy hidden states for done environments

        Args:
            obs: Current observations after environment step
            rewards: Rewards from environment step
            dones: Done flags indicating episode termination
            extras: Additional info from environment (unused)
        """
        # Update observation normalizers using the policy's update method
        # This ensures normalizer statistics stay consistent with training mode
        self.policy.update_normalization(obs)

        # Record rewards and dones in transition
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Add complete transition to storage
        self.storage.add_transition(self.transition)

        # Clear transition for next step
        self.transition.clear()

        # Reset policy for done environments (clears hidden states if recurrent)
        if dones.any():
            self.policy.reset(dones)

    def update(self) -> dict[str, float]:
        """No-op update that returns dummy metrics.

        This method:
        1. Clears storage to prevent memory buildup
        2. Updates and detaches hidden states (for recurrent policy compatibility)
        3. Returns dummy metrics for compatibility with RSL-RL runner

        Returns:
            Dictionary with dummy loss metrics (all zeros)
        """
        # Clear storage to prevent memory buildup
        self.storage.clear()

        # Update hidden states for recurrent policies
        # Detach to prevent gradient accumulation across updates
        if hasattr(self.policy, "hidden_states") and self.policy.hidden_states is not None:
            self.last_hidden_states = self.policy.hidden_states.detach()

        # Return dummy metrics that RSL-RL runner expects
        # These zeros indicate no training is occurring
        return {
            "behavior": 0.0,
            "mean_value_loss": 0.0,
            "mean_surrogate_loss": 0.0,
        }

    def compute_returns(self, obs: torch.Tensor) -> None:
        """No-op compute returns (no training occurs).

        This method is required by the RSL-RL algorithm interface but does nothing
        since TeacherPassThrough performs no training or value estimation.

        Args:
            obs: Current observations (unused in validation mode)
        """
        pass

    def broadcast_parameters(self) -> None:
        """No-op broadcast parameters for multi-GPU compatibility.

        This method is required for multi-GPU training but does nothing since
        TeacherPassThrough performs no training and thus has no parameters to sync.
        """
        pass

    def reduce_parameters(self) -> None:
        """No-op reduce parameters for multi-GPU compatibility.

        This method is required for multi-GPU training but does nothing since
        TeacherPassThrough performs no training and thus has no gradients to reduce.
        """
        pass
