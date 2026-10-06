from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REMOVED_NAMES = (
    "use_histology_spatial_head",
    "histology_dim",
    "he_projection",
)


def _active_source_files():
    roots = [
        ROOT / "deepspatial",
        ROOT / "examples",
        ROOT / "scripts",
        ROOT / "docs" / "source",
    ]
    explicit = [ROOT / "README.md", ROOT / "docs" / "morphology.md"]
    for root in roots:
        if root.is_dir():
            yield from (
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in {".py", ".md", ".toml", ".yml", ".yaml"}
            )
    yield from (path for path in explicit if path.is_file())


def test_no_removed_spatial_head_in_active_sources():
    matches = []
    for path in _active_source_files():
        text = path.read_text(encoding="utf-8")
        for name in REMOVED_NAMES:
            if name in text:
                matches.append(f"{path.relative_to(ROOT)}: {name}")

    assert not matches, "obsolete spatial-head references:\n" + "\n".join(matches)


def _is_ignored(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative_path],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def test_histology_package_is_importable():
    import deepspatial.histology as histology

    assert hasattr(histology, "FeatureStore")
    assert hasattr(histology, "HistologyConfig")


def test_local_outputs_are_ignored_but_source_is_not():
    for relative_path in (
        "data/release-boundary-probe.h5ad",
        "weights/release-boundary-probe.bin",
        "artifacts/release-boundary-probe.json",
    ):
        assert _is_ignored(relative_path), relative_path

    for relative_path in (
        "deepspatial/core.py",
        "deepspatial/histology/config.py",
        "tests/test_release_layout.py",
    ):
        assert not _is_ignored(relative_path), relative_path
