"""Download the Roboflow dataset, then sample a smaller one for the take-home.

Picks IMAGES_PER_CLASS images containing each class (an image with both counts once),
and writes them to OUTPUT_DIR as a ready-to-train YOLO dataset. Run from the project root:

    python sampling/sample-roboflow.py
"""  # noqa: E501

import os
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()  # Load environment variables from .env file

# --- What to sample ---------------------------------------------------------
CLASSES = ["mallet", "bottle"]
IMAGES_PER_CLASS = 500
VAL_FRACTION = 0.2  # share of each class's images that go to valid/
SEED = 42  # same seed -> same sample every run

DOWNLOAD_DIR = Path("data/yolov8-v2")
OUTPUT_DIR = Path("sampled")
SPLITS = ("train", "valid", "test")

# --- Download (skipped if DOWNLOAD_DIR already exists) ----------------------
rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
project = rf.workspace("malletbottle2").project("yonder-dynamics-object-detection")  # noqa: E501
version = project.version(2)
dataset = version.download("yolov8", location=str(DOWNLOAD_DIR))


def load_records(root, rng):
    """One (image, label, class ids) record per source photo, across all splits.

    Roboflow's augmentation makes several copies of each training photo, named
    '<photo>.rf.<hash>'. We keep one random copy per photo so the sample has no
    near-duplicates.
    """  # noqa: E501
    by_photo = {}
    for split in SPLITS:
        images = {p.stem: p for p in (root / split / "images").glob("*")}
        for label in sorted((root / split / "labels").glob("*.txt")):
            ids = {
                int(line.split()[0])
                for line in label.read_text().splitlines()
                if line.strip()
            }
            photo = label.stem.split(".rf.")[0]
            by_photo.setdefault(photo, []).append((images[label.stem], label, ids))  # noqa: E501
    return [rng.choice(copies) for _, copies in sorted(by_photo.items())]


def pick(records, class_ids, rng):
    """IMAGES_PER_CLASS records per class; no image is picked twice."""
    picked, used = {}, set()
    for name in CLASSES:
        pool = [r for r in records if class_ids[name] in r[2] and r[0] not in used]  # noqa: E501
        if len(pool) < IMAGES_PER_CLASS:
            raise SystemExit(
                f"Only {len(pool)} unused '{name}' images available, need {IMAGES_PER_CLASS}"  # noqa: E501
            )
        picked[name] = rng.sample(pool, IMAGES_PER_CLASS)
        used.update(r[0] for r in picked[name])
    return picked


def write_dataset(picked, names):
    """Copy the picked images and their (unchanged) labels into OUTPUT_DIR, split train/valid."""  # noqa: E501
    if OUTPUT_DIR.exists():
        if not (OUTPUT_DIR / "data.yaml").exists():
            raise SystemExit(
                f"{OUTPUT_DIR}/ exists but isn't a sampled dataset; not deleting it"  # noqa: E501
            )
        shutil.rmtree(OUTPUT_DIR)  # our own output from a previous run

    for group in picked.values():
        n_val = round(
            len(group) * VAL_FRACTION
        )  # split within each class so both appear in both splits
        for i, (image, label, _) in enumerate(group):
            split = "valid" if i < n_val else "train"
            for src, kind in ((image, "images"), (label, "labels")):
                (OUTPUT_DIR / split / kind).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, OUTPUT_DIR / split / kind / src.name)

    config = {
        "train": "train/images",
        "val": "valid/images",
        "nc": len(names),
        "names": names,
    }
    (OUTPUT_DIR / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))  # noqa: E501


def summarise(names):
    print(f"\nWrote {OUTPUT_DIR}/")
    for split in ("train", "valid"):
        labels = list((OUTPUT_DIR / split / "labels").glob("*.txt"))
        objects = Counter(
            names[int(line.split()[0])]
            for label in labels
            for line in label.read_text().splitlines()
            if line.strip()
        )
        print(f"  {split}: {len(labels)} images, objects per class {dict(objects)}")  # noqa: E501


rng = random.Random(SEED)
root = Path(dataset.location)
names = yaml.safe_load((root / "data.yaml").read_text())["names"]
class_ids = {name: names.index(name) for name in CLASSES}

records = load_records(root, rng)
print(f"{len(records)} distinct source photos in the download")
write_dataset(pick(records, class_ids, rng), names)
summarise(names)
