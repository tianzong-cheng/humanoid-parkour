"""RSL-RL distillation configuration for IRMV V3."""

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class RslRlDistillationStudentTeachersCfg(RslRlDistillationStudentTeacherCfg):
    run_path_list: list[str] = MISSING


@configclass
class IRMV_V3_FlatDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation runner configuration for IRMV V3 flat terrain with multiple teachers."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "irmv_v3_flat_distillation"
    empirical_normalization = True
    obs_groups = {
        "policy": ["student"],
        "teacher": ["teacher"],
    }
    class_name = "DistillationRunner"
    policy = RslRlDistillationStudentTeachersCfg(
        class_name="whole_body_tracking.rl.modules:StudentTeachers",
        init_noise_std=1.0,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        # Placeholder run paths - update with actual trained teacher policies
        run_path_list=[
            # Example: "your-org/humanoid-parkour/xxxxxxxx",  # motion_name
        ],
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=24,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )
