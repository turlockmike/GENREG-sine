"""
Sine Test Protein Cascade
Matching the original GENREG architecture:

Proteins process observations and produce TRUST signals that modify FITNESS.
They work IN PARALLEL with the MLP, not feeding into it.

Architecture:
  Observation ─┬─→ Controller (MLP) ─→ Output prediction
               │
               └─→ Protein Cascade ─→ Trust ─→ Fitness modifier

Both proteins and MLP neurons can exhibit saturation (hybrid computation).
"""

import numpy as np
from typing import Dict, List, Any
from . import sine_config as cfg


class Protein:
    """Base class for all proteins."""

    def __init__(self, name: str, protein_type: str):
        self.name = name
        self.protein_type = protein_type
        self.inputs: List[str] = []
        self.output = 0.0
        self.params: Dict[str, float] = {}
        self.state: Dict[str, Any] = {}

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        raise NotImplementedError

    def reset(self):
        self.output = 0.0
        for key in self.state:
            if isinstance(self.state[key], (int, float)):
                self.state[key] = 0.0

    def mutate(self, rate: float = None, scale: float = None):
        rate = rate or cfg.PROTEIN_MUTATION_RATE
        scale = scale or cfg.PROTEIN_MUTATION_SCALE

        for key, value in self.params.items():
            if np.random.random() < rate:
                new_value = value + np.random.randn() * scale
                if key == "scale":
                    new_value = np.clip(new_value, -0.2, 0.2)
                elif key == "gain":
                    new_value = np.clip(new_value, -3.0, 3.0)
                elif key == "momentum":
                    new_value = np.clip(new_value, 0.5, 0.99)
                elif key == "decay":
                    new_value = np.clip(new_value, 0.01, 0.5)
                elif key == "threshold":
                    new_value = np.clip(new_value, -10.0, 10.0)
                self.params[key] = new_value

    def clone(self) -> 'Protein':
        new_protein = self.__class__(self.name)
        new_protein.inputs = self.inputs.copy()
        new_protein.params = self.params.copy()
        new_protein.state = {k: v for k, v in self.state.items()}
        return new_protein

    def is_saturated(self, threshold: float = None) -> bool:
        threshold = threshold or cfg.SATURATION_THRESHOLD
        return abs(self.output) > threshold


class SensorProtein(Protein):
    """
    Reads and normalizes raw signals using adaptive min/max scaling.
    """

    def __init__(self, name: str = "sensor"):
        super().__init__(name, "sensor")
        self.params = {
            "gain": 1.0,
            "offset": 0.0,
        }
        self.state = {
            "running_min": float('inf'),
            "running_max": float('-inf'),
            "ema_alpha": 0.01,
        }

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        if not self.inputs:
            self.output = 0.0
            return 0.0

        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))

        alpha = self.state["ema_alpha"]
        if x < self.state["running_min"]:
            self.state["running_min"] = x
        else:
            self.state["running_min"] += alpha * (x - self.state["running_min"])

        if x > self.state["running_max"]:
            self.state["running_max"] = x
        else:
            self.state["running_max"] += alpha * (x - self.state["running_max"])

        range_val = self.state["running_max"] - self.state["running_min"]
        if range_val > 1e-6:
            normalized = 2.0 * (x - self.state["running_min"]) / range_val - 1.0
        else:
            normalized = 0.0

        self.output = np.tanh(self.params["gain"] * normalized + self.params["offset"])
        return self.output


class TrendProtein(Protein):
    """
    Detects velocity/momentum of a signal.
    """

    def __init__(self, name: str = "trend"):
        super().__init__(name, "trend")
        self.params = {
            "momentum": 0.9,
            "scale": 1.0,
        }
        self.state = {
            "prev_value": 0.0,
            "ema_delta": 0.0,
        }

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        if not self.inputs:
            self.output = 0.0
            return 0.0

        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))

        delta = x - self.state["prev_value"]
        self.state["prev_value"] = x

        momentum = self.params["momentum"]
        self.state["ema_delta"] = momentum * self.state["ema_delta"] + (1 - momentum) * delta

        self.output = np.tanh(self.params["scale"] * self.state["ema_delta"])
        return self.output


class ComparatorProtein(Protein):
    """
    Compares two inputs using different operations.
    """

    def __init__(self, name: str = "comparator"):
        super().__init__(name, "comparator")
        self.params = {
            "mode": 0,  # 0=diff, 1=ratio, 2=greater, 3=less
            "threshold": 0.0,
        }

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        if len(self.inputs) < 2:
            self.output = 0.0
            return 0.0

        def resolve(key):
            return signals.get(key, protein_outputs.get(key, 0.0))

        a = resolve(self.inputs[0])
        b = resolve(self.inputs[1])

        mode = int(self.params["mode"]) % 4

        if mode == 0:
            self.output = np.tanh(a - b)
        elif mode == 1:
            self.output = np.tanh(a / (b + 1e-6))
        elif mode == 2:
            self.output = 1.0 if a > b + self.params["threshold"] else -1.0
        else:
            self.output = 1.0 if a < b - self.params["threshold"] else -1.0

        return self.output


class IntegratorProtein(Protein):
    """
    Accumulates signal over time with decay.
    """

    def __init__(self, name: str = "integrator"):
        super().__init__(name, "integrator")
        self.params = {
            "decay": 0.05,
            "scale": 1.0,
        }
        self.state = {
            "accumulator": 0.0,
        }

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        if not self.inputs:
            self.output = 0.0
            return 0.0

        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))

        decay = max(0.001, min(0.5, self.params["decay"]))
        self.state["accumulator"] = self.state["accumulator"] * (1 - decay) + x

        self.output = np.tanh(self.params["scale"] * self.state["accumulator"])
        return self.output


class GateProtein(Protein):
    """
    Conditionally activates based on threshold with hysteresis.
    """

    def __init__(self, name: str = "gate"):
        super().__init__(name, "gate")
        self.params = {
            "threshold": 0.0,
            "hysteresis": 0.1,
        }
        self.state = {
            "is_open": False,
        }

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        if len(self.inputs) < 2:
            self.output = 0.0
            return 0.0

        def resolve(key):
            return signals.get(key, protein_outputs.get(key, 0.0))

        condition = resolve(self.inputs[0])
        signal = resolve(self.inputs[1])

        thresh = self.params["threshold"]
        hyst = self.params["hysteresis"]

        if self.state["is_open"]:
            if condition < thresh - hyst:
                self.state["is_open"] = False
        else:
            if condition > thresh + hyst:
                self.state["is_open"] = True

        self.output = signal if self.state["is_open"] else 0.0
        return self.output


class TrustModifierProtein(Protein):
    """
    Converts protein output into trust delta.
    THE BRIDGE between perception and selection pressure.
    This is what makes proteins affect fitness!
    """

    def __init__(self, name: str = "trust_mod"):
        super().__init__(name, "trust_modifier")
        self.params = {
            "scale": 0.1,
            "gain": 1.0,
            "bias": 0.0,
        }

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        if not self.inputs:
            self.output = 0.0
            return 0.0

        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))

        self.output = self.params["scale"] * np.tanh(self.params["gain"] * x + self.params["bias"])
        return self.output


def create_sine_protein_cascade() -> List[Protein]:
    """
    Create protein cascade for sine approximation.

    Proteins detect patterns in the input and produce TRUST signals
    that modify fitness. This creates selection pressure for good behavior.

    For sine: trust should reward accurate tracking of the sine wave pattern.
    """
    proteins = []

    # =====================================================
    # LAYER 1: Sensors - normalize and process raw input
    # =====================================================

    # Main input sensor
    sensor_x = SensorProtein("sensor_x")
    sensor_x.inputs = ["x"]
    proteins.append(sensor_x)

    # Sensor for prediction (to compare with pattern)
    sensor_pred = SensorProtein("sensor_pred")
    sensor_pred.inputs = ["prediction"]
    proteins.append(sensor_pred)

    # Sensor for target (true sine value)
    sensor_target = SensorProtein("sensor_target")
    sensor_target.inputs = ["target"]
    proteins.append(sensor_target)

    # =====================================================
    # LAYER 2: Trend detection - are we tracking correctly?
    # =====================================================

    # Trend of input (is x increasing/decreasing?)
    trend_x = TrendProtein("trend_x")
    trend_x.inputs = ["sensor_x"]
    proteins.append(trend_x)

    # Trend of prediction
    trend_pred = TrendProtein("trend_pred")
    trend_pred.inputs = ["sensor_pred"]
    proteins.append(trend_pred)

    # Trend of target
    trend_target = TrendProtein("trend_target")
    trend_target.inputs = ["sensor_target"]
    proteins.append(trend_target)

    # =====================================================
    # LAYER 3: Comparators - how close is prediction to target?
    # =====================================================

    # Compare prediction to target (error signal)
    error_comp = ComparatorProtein("error_comparator")
    error_comp.inputs = ["sensor_pred", "sensor_target"]
    error_comp.params["mode"] = 0  # difference
    proteins.append(error_comp)

    # Compare prediction trend to target trend (derivative matching)
    trend_comp = ComparatorProtein("trend_comparator")
    trend_comp.inputs = ["trend_pred", "trend_target"]
    trend_comp.params["mode"] = 0
    proteins.append(trend_comp)

    # =====================================================
    # LAYER 4: Integrators - accumulate patterns over time
    # =====================================================

    # Integrate error (sustained error is bad)
    error_integrator = IntegratorProtein("error_integrator")
    error_integrator.inputs = ["error_comparator"]
    error_integrator.params["decay"] = 0.1
    proteins.append(error_integrator)

    # Integrate trend match (sustained good tracking)
    trend_integrator = IntegratorProtein("trend_integrator")
    trend_integrator.inputs = ["trend_comparator"]
    trend_integrator.params["decay"] = 0.05
    proteins.append(trend_integrator)

    # =====================================================
    # LAYER 5: Trust Modifiers - convert signals to fitness
    # =====================================================

    # Penalize prediction error (larger error = negative trust)
    trust_error = TrustModifierProtein("trust_error")
    trust_error.inputs = ["error_integrator"]
    trust_error.params["scale"] = -0.1  # Negative: error reduces trust
    trust_error.params["gain"] = 2.0
    proteins.append(trust_error)

    # Reward trend matching (good derivative = positive trust)
    trust_trend = TrustModifierProtein("trust_trend")
    trust_trend.inputs = ["trend_integrator"]
    trust_trend.params["scale"] = 0.05  # Positive: matching boosts trust
    trust_trend.params["gain"] = 1.0
    proteins.append(trust_trend)

    # Reward low instantaneous error
    trust_instant = TrustModifierProtein("trust_instant")
    trust_instant.inputs = ["error_comparator"]
    trust_instant.params["scale"] = -0.05  # Small penalty for instant error
    trust_instant.params["gain"] = 1.0
    proteins.append(trust_instant)

    return proteins


def run_protein_cascade(
    proteins: List[Protein],
    x: float,
    prediction: float,
    target: float
) -> float:
    """
    Run the protein cascade.
    Returns total trust delta from all TrustModifierProteins.

    Args:
        proteins: List of Protein objects
        x: Raw input value
        prediction: MLP's prediction
        target: True sin(x) value

    Returns:
        trust_delta: Sum of all TrustModifierProtein outputs
    """
    # Build signals dict
    signals = {
        "x": x,
        "prediction": prediction,
        "target": target,
    }

    # Run cascade - pre-allocate output dict with expected size
    protein_outputs = {}
    trust_delta = 0.0

    # Cache protein type checks to avoid repeated string comparisons
    for protein in proteins:
        output = protein.forward(signals, protein_outputs)
        protein_outputs[protein.name] = output

    # Sum trust modifiers in a separate pass (allows for potential future vectorization)
    for protein in proteins:
        if protein.protein_type == "trust_modifier":
            trust_delta += protein.output

    return trust_delta


def reset_protein_cascade(proteins: List[Protein]):
    """Reset all proteins' internal state."""
    for protein in proteins:
        protein.reset()


def get_cascade_saturation_stats(proteins: List[Protein]) -> Dict:
    """Get saturation statistics for the protein cascade."""
    saturated_count = 0
    saturation_values = []

    for protein in proteins:
        sat = abs(protein.output)
        saturation_values.append(sat)
        if protein.is_saturated():
            saturated_count += 1

    return {
        'k_proteins': saturated_count,
        'n_proteins': len(proteins),
        'per_protein_saturation': saturation_values,
        'mean_saturation': np.mean(saturation_values) if saturation_values else 0,
    }
