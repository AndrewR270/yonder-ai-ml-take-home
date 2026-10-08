"""Upload the sampled dataset (sampled/) to YOUR OWN Roboflow project. Run from the project root:

    python sampling/upload-roboflow.py             upload
    python sampling/upload-roboflow.py --dry-run   show where it would upload, without uploading

Create an empty object-detection project in Roboflow first, then set these in .env
(see .env.example). Nothing here is hard-coded, so you only ever upload to your own project.
"""  # noqa: E501

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()  # Load environment variables from .env file

DATASET_DIR = Path("sampled")


def require(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing {name}. Set it in .env (see .env.example).")
    return value


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="show where it would upload, without uploading",
)
parser.add_argument(
    "--allow-existing",
    action="store_true",
    help="upload even if the project already has images",
)
args = parser.parse_args()

api_key = require("PROJECT_API_KEY")
workspace_name = require("ROBOFLOW_WORKSPACE")
project_name = require("ROBOFLOW_PROJECT")

if not (DATASET_DIR / "data.yaml").exists():
    raise SystemExit(
        f"No dataset in {DATASET_DIR}/. Run sampling/sample-roboflow.py first."
    )
n_images = sum(
    len(list((DATASET_DIR / split / "images").glob("*")))
    for split in ("train", "valid")
)

workspace = Roboflow(api_key=api_key).workspace(workspace_name)
try:
    project = workspace.project(project_name)
except (
    Exception
) as e:  # not printing the error itself: it can contain the API key # noqa: E501
    raise SystemExit(
        f"Couldn't open project '{workspace_name}/{project_name}' ({type(e).__name__}). "  # noqa: E501
        "Check ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT and PROJECT_API_KEY, and that the project exists."  # noqa: E501
    )

print(f"{n_images} images -> {project.id} (currently {project.images} images)")
if project.images and not args.allow_existing:
    raise SystemExit(
        "That project already has images. Use a new empty project, or pass --allow-existing."  # noqa: E501
    )
if args.dry_run:
    raise SystemExit("Dry run: nothing uploaded.")

workspace.upload_dataset(
    str(DATASET_DIR),
    project_name,
    project_type=project.type,
    batch_name="sample-500-per-class",
    num_workers=10,
    num_retries=2,
)
print("Upload finished.")
