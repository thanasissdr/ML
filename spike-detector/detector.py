from typing import Optional, List, Protocol

import numpy as np
import pandas as pd
from dataclasses import dataclass


class SpikeDetector(Protocol):
    def run(self, value: float) -> bool:
        ...


@dataclass
class SpikeDetectorFixed:
    """
    If a value is higher/lower to specific thresholds
    mark it as a spike.
    """

    upper_bound: float
    lower_bound: Optional[float] = None

    def run(self, value: float) -> bool:
        is_upper_outlier = self.is_upper_outlier(value)

        is_lower_outlier = False
        if self.lower_bound:
            is_lower_outlier = self.is_lower_outlier(value)

        is_outlier = any([is_lower_outlier, is_upper_outlier])
        return is_outlier

    def is_upper_outlier(self, value: float) -> bool:
        return self.upper_bound < value

    def is_lower_outlier(self, value: float) -> bool:
        return self.lower_bound > value


class NotEnoughDataException(Exception):
    pass


@dataclass
class SpikeDetectorMovingAverage:
    """
    Checks if a value could be considered an outlier or not using
    a rolling window approach. Let rolling_mean and rolling_std be
    the mean and standard deviation of the previous window:

    If rolling_mean + (-) threshold * rolling_std < (>) new_value then
    the value is considered an outlier.
    """

    window: int = 3
    threshold: float = 3.0
    detect_sag: bool = True

    def run(self, value: float, ddof: int = 0) -> bool:
        mean = self.rolling_window_states.mean()
        std = self.rolling_window_states.std(ddof=ddof)

        is_outlier = self.is_outlier(mean, std, value)

        # Update rolling window states
        self.update_rolling_window(value)

        return is_outlier

    def is_outlier(self, mean: float, std: float, value: float) -> bool:
        is_upper_outlier = self.is_upper_outlier(mean, std, value)

        is_lower_outlier = False
        if self.detect_sag:
            is_lower_outlier = self.is_lower_outlier(mean, std, value)

        is_outlier = any([is_lower_outlier, is_upper_outlier])
        return is_outlier

    def is_upper_outlier(self, mean: float, std: float, value: float) -> bool:
        return (mean + self.threshold * std) < value

    def is_lower_outlier(self, mean: float, std: float, value: float) -> bool:
        return (mean - self.threshold * std) > value

    def set_initial_window_states(self, data: List | np.ndarray):
        if len(data) > self.window:
            self.rolling_window_states = data[: self.window]

        else:
            raise NotEnoughDataException(
                f"Length of initial data should be at least equal to the size of the window + 1, i.e. {self.window + 1}"
            )

    def update_rolling_window(self, value: float) -> None:
        self.rolling_window_states = np.delete(self.rolling_window_states, 0)
        self.rolling_window_states = np.append(self.rolling_window_states, value)
