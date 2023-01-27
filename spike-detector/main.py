from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field

from detector import SpikeDetector, SpikeDetectorMovingAverage



ListLike = Union[np.ndarray, List]


@dataclass
class SpikeDetectorRunner:
    spike_detector: SpikeDetector
    states: List = field(default_factory=lambda: [])
        
    def run(self, values: ListLike, ax = None, color='red', marker='x', **kwargs):
        spikes = self.get_spikes(values)
        self.plot(spikes, ax, color, marker, **kwargs)
        

    def get_spikes(self, values: ListLike) -> np.ndarray:
        if isinstance(self.spike_detector, (SpikeDetectorMovingAverage,)):
            self.spike_detector.set_initial_window_states(values)

        outliers = []
        for v in values:
            self.states = np.append(self.states, v)
            is_outlier = self.spike_detector.run(v)
            outliers.append(is_outlier)

        return np.array(outliers)
    
    
    def plot(self, spikes: np.ndarray,  ax = None, color='red', marker='x', **kwargs):
        spikes_mask = np.where(spikes == True)

        if not ax:
            _, ax = plt.subplots(1, 1, figsize=(16, 8))

        ax.plot(self.states)
        ax.scatter(
            spikes_mask, self.states[spikes_mask], color=color, marker=marker, **kwargs
        )

        return ax