from __future__ import annotations

import collections

import numpy as np

from river import base, optim, utils

__all__ = ["MultiOutputPARegressor"]

class MultiOutputBasePA:
    def __init__(self, C, mode, learn_intercept, num_outputs):
        self.C = C
        self.mode = mode
        self.calc_tau = {0: self._calc_tau_0, 1: self._calc_tau_1, 2: self._calc_tau_2}[mode]
        self.learn_intercept = learn_intercept
        self.weights = [collections.defaultdict(float) for _ in range(num_outputs)]
        self.intercepts = np.zeros(num_outputs)
        self.num_outputs = num_outputs

    @classmethod
    def _calc_tau_0(cls, x, losses):
        norm = utils.math.norm(x, order=2) ** 2
        taus = {}
        for output, loss in losses.items():
            taus[output] = loss / norm if norm > 0 else 0
        return taus

    def _calc_tau_1(self, x, losses):
        norm = utils.math.norm(x, order=2) ** 2
        taus = {}
        for output, loss in losses.items():
            taus[output] = min(self.C, loss / norm) if norm > 0 else 0
        return taus

    def _calc_tau_2(self, x, losses):
        taus = {}
        for output, loss in losses.items():
            taus[output] = loss / (utils.math.norm(x, order=2) ** 2 + 0.5 / self.C)
        return taus

class MultiOutputPARegressor(MultiOutputBasePA, base.MultiTargetRegressor):
    """MultiOutput Passive-aggressive learning for regression.

    Parameters
    ----------
    C
    mode
    eps
    learn_intercept
    num_outputs

    """

    def __init__(self, C=1.0, mode=1, eps=0.1, learn_intercept=True, num_outputs=6):
        super().__init__(C=C, mode=mode, learn_intercept=learn_intercept, num_outputs=num_outputs)
        self.eps = eps
        self.loss = optim.losses.MultiOutputEpsilonInsensitiveHinge(eps=eps)

    def learn_one(self, x, y):
        y_pred = self.predict_one(x)

        losses = {output: self.loss(y[output], y_pred.get(output, 0)) for output in y.keys()}
        taus = self.calc_tau(x, losses)
        
        for output in range(self.num_outputs):
            if output in y:  # if there's a target for this output
                step = taus[output] * np.sign(y[output] - y_pred.get(output, 0))
                for i, xi in x.items():
                    self.weights[output][i] += step * xi
                if self.learn_intercept:
                    self.intercepts[output] += step
        return self


    def predict_one(self, x):
        y_pred = {}
        for i in range(self.num_outputs):
            y_pred[i] = utils.math.dot(x, self.weights[i]) + self.intercepts[i]
        return y_pred


