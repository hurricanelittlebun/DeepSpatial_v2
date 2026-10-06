from dataclasses import dataclass, asdict
import math


@dataclass
class HistologyConfig:
    """One registered tissue/g series per store; all physical values in um."""

    use_histology: bool = False
    use_morphology_uot: bool = False
    use_morphology_path: bool = False
    use_nucleus_path: bool = False
    feature_path: str | None = None
    nucleus_feature_path: str | None = None
    path_cache: str | None = None
    section_key: str = "section_id"
    coordinate_frame: str | None = None
    spatial_weight: float | None = None
    gene_weight: float | None = None
    histology_weight: float = 1.0
    class_weight: float = 10.0
    spatial_metric: str = "legacy_normalized"
    candidate_radius_um: float = 100.0
    candidate_spacing_um: float = 25.0
    move_weight: float = 0.001
    morphology_weight: float = 1.0
    nucleus_weight: float = 0.0
    max_candidates: int = 1024
    allow_linear_fallback: bool = False
    max_uot_entries: int = 25_000_000
    feature_cache_mb: int = 512
    seed: int = 0

    def __post_init__(self):
        if self.spatial_metric not in ("legacy_normalized", "physical_um"):
            raise ValueError("spatial_metric must be legacy_normalized or physical_um")
        if not self.use_histology and any(
            (
                self.use_morphology_uot,
                self.use_morphology_path,
                self.use_nucleus_path,
            )
        ):
            raise ValueError("Histology subcomponents require use_histology=True")
        if self.use_histology and (not self.feature_path or not self.coordinate_frame):
            raise ValueError(
                "Histology requires feature_path and explicit coordinate_frame"
            )
        if self.use_morphology_path and not self.path_cache:
            raise ValueError("Morphology path requires a persistent path_cache")
        if self.use_nucleus_path and not self.use_morphology_path:
            raise ValueError("Nucleus path requires use_morphology_path=True")
        if self.use_nucleus_path and not self.nucleus_feature_path:
            raise ValueError("Nucleus path requires nucleus_feature_path")
        for key in (
            "histology_weight",
            "class_weight",
            "move_weight",
            "morphology_weight",
            "nucleus_weight",
            "candidate_radius_um",
        ):
            v = getattr(self, key)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("spatial_weight", "gene_weight"):
            v = getattr(self, key)
            if v is not None and (not math.isfinite(v) or v < 0):
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in (
            "candidate_spacing_um",
            "max_candidates",
            "max_uot_entries",
            "feature_cache_mb",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")

    def to_dict(self):
        return asdict(self)
