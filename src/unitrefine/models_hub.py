"""
Downloads UnitRefine models from the Hugging Face Hub. Used by the GUI (Load models from Hugging Face Hub) and
by unitrefine-download-models (see cli.py), so has no Qt.
"""

import json
import re
import shutil
from pathlib import Path

# UnitRefine models published on the Hugging Face Hub, shown as options when loading from HFH.
# Each group is shown with a gap before it and an optional heading; models are (repo id, note shown after the name).
CASCADE_NOTE = "(can be combined in a cascade for good / MUA / noise labels)"
LIGHTWEIGHT_NOTE = "(based on fast-to-compute metrics, no PCA metrics; can be combined in a cascade for good / MUA / noise labels)"
hfh_model_groups = [
    dict(title=None, models=[
        ("SpikeInterface/UnitRefine_noise_neural_classifier", CASCADE_NOTE),
        ("SpikeInterface/UnitRefine_sua_mua_classifier", CASCADE_NOTE),
    ]),
    dict(title=None, models=[
        ("SpikeInterface/UnitRefine_noise_neural_classifier_lightweight", LIGHTWEIGHT_NOTE),
        ("SpikeInterface/UnitRefine_sua_mua_classifier_lightweight", LIGHTWEIGHT_NOTE),
    ]),
    dict(title="SUA only models", models=[
        ("AnoushkaJain3/UnitRefine-generalized-sua-classifier", ""),
        ("AnoushkaJain3/UnitRefine-mice-sua-classifier", ""),
        ("AnoushkaJain3/UnitRefine-rat-sua-classifier", ""),
        ("AnoushkaJain3/UnitRefine-mole-rat-sua-classifier", ""),
        ("AnoushkaJain3/UnitRefine-monkey-sua-classifier", ""),
        ("AnoushkaJain3/UnitRefine-human-sua-classifier", ""),
    ]),
]


# the file in a downloaded model's folder which says which Hugging Face repo it came from
HFH_PATH_FILE = "hfh_path.txt"

# class names used by models on the Hugging Face Hub, as the GUI labels them
LABEL_TRANSLATION = {
    'MUA': ['mua'],
    'good': ['sua', 'SUA', 'neural'],
    'noise': [],
}


def looks_like_repo_id(text):
    """True for a Hugging Face repo id, e.g. SpikeInterface/UnitRefine_sua_mua_classifier."""
    return re.fullmatch(r"[\w.-]+/[\w.-]+", str(text)) is not None


def downloaded_repo_id(model_folder):
    """The Hugging Face repo a model folder was downloaded from, or None if it wasn't."""
    hfh_path = Path(model_folder) / HFH_PATH_FILE
    return hfh_path.read_text().strip() if hfh_path.is_file() else None


def translate_label_conversion(model_info_path):
    """Renames the classes in a downloaded model's model_info.json to the GUI's labels, e.g. sua to good."""
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

    updated_model_info = {}
    for label_index, label in model_info['label_conversion'].items():
        new_label = label
        for sigui_label, user_labels in LABEL_TRANSLATION.items():
            if label in user_labels:
                new_label = sigui_label
        updated_model_info[label_index] = new_label
    model_info['label_conversion'] = updated_model_info

    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=4)


def download_model(repo_id, model_directory):
    """
    Downloads the model in the Hugging Face repo `repo_id` into `model_directory`, which must not exist yet,
    and renames its classes to the GUI's labels. Nothing is left behind if the download fails.
    """
    from huggingface_hub import snapshot_download

    model_directory = Path(model_directory)
    print(f"Downloading model from HuggingFaceHub repo {repo_id}.\n")
    try:
        snapshot_download(repo_id, local_dir=model_directory)
        (model_directory / HFH_PATH_FILE).write_text(repo_id)
        translate_label_conversion(model_directory / "model_info.json")
    except BaseException:
        # don't leave a half-downloaded model behind
        shutil.rmtree(model_directory, ignore_errors=True)
        raise
