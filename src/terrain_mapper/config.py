from dataclasses import dataclass, field

@dataclass
class HardwareConfig:
    """Hardware-in-the-Loop (HIL) configuration for live sensor data."""
    use_hardware: bool = False
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 115200

@dataclass(frozen=True)
class PlannerConfig:
    """Thresholds for converting terrain slope into traversal costs."""

    max_slope_degrees: float = 30.0
    slope_cost_weight: float = 4.0
    minimum_cost: float = 1.0
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def __post_init__(self) -> None:
        if not 0 < self.max_slope_degrees < 90:
            raise ValueError("max_slope_degrees must be between 0 and 90")
        if self.slope_cost_weight < 0:
            raise ValueError("slope_cost_weight cannot be negative")
        if self.minimum_cost <= 0:
            raise ValueError("minimum_cost must be positive")