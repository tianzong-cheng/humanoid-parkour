"""Reorder joints and bodies in an npz motion file to a standard ordering.

This script loads an npz file containing motion data with joint and body names,
extracts the last 29 joints (matching joint_names) and all bodies (matching body_names),
reorders them to a predefined standard order, and saves a new npz file with only the
essential arrays.

Output is automatically saved to artifacts/reordered/{input_filename} where
{input_filename} is the basename of the input file.

Input npz format must contain:
    ['fps', 'joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
     'body_lin_vel_w', 'body_ang_vel_w', 'joint_names', 'body_names']

Output npz format:
    ['fps', 'joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
     'body_lin_vel_w', 'body_ang_vel_w']

Joint ordering (29 joints):
    ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
     'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
     'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
     'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint',
     'right_shoulder_pitch_joint', 'left_ankle_pitch_joint',
     'right_ankle_pitch_joint', 'left_shoulder_roll_joint',
     'right_shoulder_roll_joint', 'left_ankle_roll_joint',
     'right_ankle_roll_joint', 'left_shoulder_yaw_joint',
     'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint',
     'left_wrist_roll_joint', 'right_wrist_roll_joint',
     'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
     'left_wrist_yaw_joint', 'right_wrist_yaw_joint']

Body ordering (30 bodies):
    ['pelvis', 'left_hip_pitch_link', 'right_hip_pitch_link', 'waist_yaw_link',
     'left_hip_roll_link', 'right_hip_roll_link', 'waist_roll_link',
     'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link',
     'left_knee_link', 'right_knee_link', 'left_shoulder_pitch_link',
     'right_shoulder_pitch_link', 'left_ankle_pitch_link',
     'right_ankle_pitch_link', 'left_shoulder_roll_link',
     'right_shoulder_roll_link', 'left_ankle_roll_link',
     'right_ankle_roll_link', 'left_shoulder_yaw_link',
     'right_shoulder_yaw_link', 'left_elbow_link', 'right_elbow_link',
     'left_wrist_roll_link', 'right_wrist_roll_link',
     'left_wrist_pitch_link', 'right_wrist_pitch_link',
     'left_wrist_yaw_link', 'right_wrist_yaw_link']
"""

import argparse
import numpy as np
from pathlib import Path

TARGET_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

TARGET_BODY_ORDER = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


def load_and_reorder(input_path: str):
    """Load npz, reorder joints and bodies, save to output.

    Output is saved to artifacts/reordered/{input_filename} where input_filename
    is the basename of the input file.
    """
    # Generate output path
    input_path_obj = Path(input_path)
    output_dir = Path("artifacts/reordered")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path_obj.name
    data = np.load(input_path, allow_pickle=True)
    # Ensure all required keys exist
    required_keys = [
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "body_names",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Input npz missing required key: {key}")

    fps = data["fps"]
    joint_pos = data["joint_pos"]  # shape (time, num_joints)
    joint_vel = data["joint_vel"]  # shape (time, num_joints)
    body_pos_w = data["body_pos_w"]  # shape (time, num_bodies, 3)
    body_quat_w = data["body_quat_w"]  # shape (time, num_bodies, 4)
    body_lin_vel_w = data["body_lin_vel_w"]  # shape (time, num_bodies, 3)
    body_ang_vel_w = data["body_ang_vel_w"]  # shape (time, num_bodies, 3)
    joint_names = data["joint_names"]  # shape (num_joints,) or (num_joints,)
    body_names = data["body_names"]  # shape (num_bodies,)

    # Convert object arrays to strings if needed (allow_pickle may keep them as objects)
    joint_names = [str(name) for name in joint_names]
    body_names = [str(name) for name in body_names]

    joint_pos_last = joint_pos[:, -29:]
    joint_vel_last = joint_vel[:, -29:]

    # Create mapping from target joint name to index in joint_names
    joint_name_to_idx = {name: i for i, name in enumerate(joint_names)}
    # Build reordering index list: for each target joint, find its position in joint_names
    joint_reorder_idx = []
    missing_joints = []
    for target in TARGET_JOINT_ORDER:
        if target in joint_name_to_idx:
            joint_reorder_idx.append(joint_name_to_idx[target])
        else:
            missing_joints.append(target)
    if missing_joints:
        raise ValueError(f"Target joint(s) not found in input joint_names: {missing_joints}")
    # Apply reordering
    joint_pos_reordered = joint_pos_last[:, joint_reorder_idx]
    joint_vel_reordered = joint_vel_last[:, joint_reorder_idx]

    # Body mapping
    body_name_to_idx = {name: i for i, name in enumerate(body_names)}
    body_reorder_idx = []
    missing_bodies = []
    for target in TARGET_BODY_ORDER:
        if target in body_name_to_idx:
            body_reorder_idx.append(body_name_to_idx[target])
        else:
            missing_bodies.append(target)
    if missing_bodies:
        raise ValueError(f"Target body(s) not found in input body_names: {missing_bodies}")
    # Apply reordering to each body array (axis=1)
    body_pos_w_reordered = body_pos_w[:, body_reorder_idx, :]
    body_quat_w_reordered = body_quat_w[:, body_reorder_idx, :]
    body_lin_vel_w_reordered = body_lin_vel_w[:, body_reorder_idx, :]
    body_ang_vel_w_reordered = body_ang_vel_w[:, body_reorder_idx, :]

    # Save output npz
    np.savez(
        output_path,
        fps=fps,
        joint_pos=joint_pos_reordered,
        joint_vel=joint_vel_reordered,
        body_pos_w=body_pos_w_reordered,
        body_quat_w=body_quat_w_reordered,
        body_lin_vel_w=body_lin_vel_w_reordered,
        body_ang_vel_w=body_ang_vel_w_reordered,
    )

    print(f"Successfully reordered and saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Reorder joints and bodies in an npz motion file to standard ordering."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input npz file")
    args = parser.parse_args()

    output_path = load_and_reorder(args.input)

    # Upload to wandb using input filename as collection name
    import wandb

    COLLECTION = Path(args.input).stem
    run = wandb.init(project="reorder_npz", name=COLLECTION)
    print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
    REGISTRY = "Motions"
    logged_artifact = run.log_artifact(artifact_or_path=str(output_path), name=COLLECTION, type=REGISTRY)
    run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
    print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")


if __name__ == "__main__":
    main()
