import sys
from contextlib import suppress

import wandb

WANDB_ENTITY = "tianzong-cheng-shanghai-jiao-tong-university"
WANDB_PROJECT = "humanoid-parkour"
RUN_ID = ""
FILE_PATH = ""
ARTIFACT_NAME = ""
ARTIFACT_TYPE = "skillset"


def validate_run(entity: str, project: str, run_id: str) -> bool:
    run_path = f"{entity}/{project}/{run_id}"
    try:
        api = wandb.Api()
        api.run(run_path)
        return True
    except Exception as e:
        print(f"Error validating run: {e}")
        return False


def delete_previous_artifacts(entity: str, project: str, run_id: str, artifact_type: str) -> None:
    with suppress(Exception):
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        artifacts_to_delete = []
        for artifact in run.logged_artifacts():
            if artifact.type == artifact_type:
                artifacts_to_delete.append(artifact)
        if artifacts_to_delete:
            for artifact in artifacts_to_delete:
                with suppress(Exception):
                    artifact.delete(delete_aliases=True)


def upload_artifact(
    entity: str, project: str, run_id: str, file_path: str, artifact_name: str, artifact_type: str
) -> bool:
    try:
        run = wandb.init(entity=entity, project=project, id=run_id, resume="allow")
        logged_artifact = run.log_artifact(artifact_or_path=file_path, name=artifact_name, type=artifact_type)
        logged_artifact.wait()
        print("Uploaded successfully")
        run.finish()
        return True
    except Exception as e:
        print(f"Failed to upload: {e}")
        return False


def main():
    if not validate_run(WANDB_ENTITY, WANDB_PROJECT, RUN_ID):
        sys.exit(1)
    delete_previous_artifacts(WANDB_ENTITY, WANDB_PROJECT, RUN_ID, ARTIFACT_TYPE)
    if not upload_artifact(WANDB_ENTITY, WANDB_PROJECT, RUN_ID, FILE_PATH, ARTIFACT_NAME, ARTIFACT_TYPE):
        sys.exit(1)


if __name__ == "__main__":
    main()
