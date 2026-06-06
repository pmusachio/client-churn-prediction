from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


def test_project_config_is_valid_toml():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "project.toml"
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    assert config["project"]["name"]
    assert config["project"]["problem_type"]
    assert "data" in config


def test_standard_directories_exist():
    root = Path(__file__).resolve().parents[1]
    for relative in ["data/raw", "data/interim", "data/processed", "models", "reports", "src"]:
        assert (root / relative).exists()
