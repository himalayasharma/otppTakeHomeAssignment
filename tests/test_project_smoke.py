from pathlib import Path
import sys


def test_package_import_and_expected_artifacts_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import src

    assert src is not None
    assert (repo_root / "SPEC.md").exists()
    assert (repo_root / "EDA_FINDINGS.md").exists()
    assert (repo_root / "notebooks" / "01_eda.ipynb").exists()
