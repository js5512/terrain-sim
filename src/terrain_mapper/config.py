from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerConfig:
    """Thresholds for converting terrain slope into traversal costs."""

    max_slope_degrees: float = 30.0
    slope_cost_weight: float = 4.0
    minimum_cost: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.max_slope_degrees < 90:
            raise ValueError("max_slope_degrees must be between 0 and 90")
        if self.slope_cost_weight < 0:
            raise ValueError("slope_cost_weight cannot be negative")
        if self.minimum_cost <= 0:
            raise ValueError("minimum_cost must be positive")