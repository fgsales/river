from __future__ import annotations

from river import stream

from . import base


class METR_LA(base.FileDataset):
    """Solar flare multi-output regression.

    References
    ----------
    [^1]: [UCI page](https://archive.ics.uci.edu/ml/datasets/Solar+Flare)

    """

    def __init__(self):
        self.past_history = 12
        self.forecast_horizon = 6
        super().__init__(
            task=base.MO_REG,
            filename="data.csv",
            directory="data/PEMS-BAY",
            n_features=325,
            n_samples=52116-self.past_history-self.forecast_horizon,
        )

    def __iter__(self):
        return stream.sliding_window_iter_csv(
            self.path,
            drop=['datetime'],
            past_history=self.past_history,
            forecast_horizon=self.forecast_horizon,
        )
