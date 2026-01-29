import torch


def task_selector(num_envs: int, num_tasks: int, device: str) -> torch.Tensor:
    selector = torch.zeros(num_envs, dtype=torch.long, device=device)
    envs_per_teacher = num_envs // num_tasks
    remainder = num_envs % num_tasks
    start_idx = 0
    for task_idx in range(num_tasks):
        end_idx = start_idx + envs_per_teacher + (1 if task_idx < remainder else 0)
        selector[start_idx:end_idx] = task_idx
        start_idx = end_idx
    return selector
