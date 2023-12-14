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
        super().__init__(
            task=base.MO_REG,
            filename="data.csv",
            directory="data/METR-LA",
            n_features=207,
            n_samples=34272,
        )
        self.past_history = 12
        self.forecast_horizon = 6

    def __iter__(self):
        return stream.sliding_window_iter_csv(
            self.path,
            drop=['datetime'],
            past_history=self.past_history,
            forecast_horizon=self.forecast_horizon,
        )
