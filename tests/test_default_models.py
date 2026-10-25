"""A new project comes with the pretrained models from the Hugging Face Hub, chosen as a cascade."""

import shutil
from pathlib import Path

import pytest

from unitrefine import models_hub

# the trained test models stand in for the Hugging Face repos
FAKE_HUB = {
    "hub/noise_neural": "test_model_noise_vs_neural",
    "hub/sua_mua": "test_model_sua_vs_mua",
    "hub/sua": "test_model_sua_vs_all",
}


@pytest.fixture
def fake_hub(trained_project, monkeypatch):
    """Downloads copy the trained test models; returns the repo ids downloaded, and those which fail."""
    hub = dict(downloaded=[], offline=set())

    def download_model(repo_id, model_directory):
        if repo_id in hub["offline"]:
            raise ConnectionError("no internet")
        hub["downloaded"].append(repo_id)
        shutil.copytree(Path(trained_project) / "models" / FAKE_HUB[repo_id], model_directory)
        (Path(model_directory) / models_hub.HFH_PATH_FILE).write_text(repo_id)

    monkeypatch.setattr(models_hub, "download_model", download_model)
    return hub


def new_project(folder, hub):
    from unitrefine.main import UnitRefineProject, load_project

    UnitRefineProject(folder).save()
    failed = models_hub.download_default_models(folder / "models", list(FAKE_HUB))
    return load_project(folder), failed


def test_a_new_project_comes_with_the_models_as_a_cascade(fake_hub, tmp_path, qapp):
    from unitrefine.launcher import choose_pretrained_cascade
    from unitrefine.main import MainWindow

    project, failed = new_project(tmp_path / "project", fake_hub)
    assert failed == {}
    assert sorted(Path(folder).name for folder, source in project.model_paths if source == "hfh") == ["noise_neural", "sua", "sua_mua"]

    window = MainWindow(project)
    choose_pretrained_cascade(window)
    selector = window.model_selector
    assert selector.mode() == "cascade" and selector.describe_problem() is None
    assert [folder.name for folder, _ in selector.selected_models()] == ["noise_neural", "sua_mua"]
    window.close()


def test_without_internet_the_project_opens_without_them(fake_hub, tmp_path, qapp):
    from unitrefine.launcher import choose_pretrained_cascade
    from unitrefine.main import MainWindow

    fake_hub["offline"] = {"hub/sua_mua"}
    project, failed = new_project(tmp_path / "project", fake_hub)
    assert list(failed) == ["hub/sua_mua"] and "no internet" in failed["hub/sua_mua"]
    # nothing half downloaded is left behind
    assert sorted(path.name for path in (tmp_path / "project" / "models").iterdir()) == ["noise_neural", "sua"]

    window = MainWindow(project)
    choose_pretrained_cascade(window)
    # a cascade needs both layers: the models are offered one by one instead
    assert window.model_selector.mode() == "binary" and window.model_selector.describe_problem() is None
    window.close()


def test_models_already_in_the_project_are_not_downloaded_again(fake_hub, tmp_path):
    new_project(tmp_path / "project", fake_hub)
    fake_hub["downloaded"].clear()
    assert models_hub.download_default_models(tmp_path / "project" / "models", list(FAKE_HUB)) == {}
    assert fake_hub["downloaded"] == []


def test_the_default_models_are_published_models():
    published = [repo_id for group in models_hub.hfh_model_groups for repo_id, _ in group["models"]]
    assert set(models_hub.DEFAULT_MODELS) <= set(published)
