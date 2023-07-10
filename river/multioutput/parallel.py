from __future__ import annotations

import collections
import copy
import functools
import itertools
import random

from river import base, linear_model
from river.utils.math import minkowski_distance, prod

__all__ = [
    "RegressorParallel",
]


class BaseParallel(base.Wrapper, collections.UserDict):
    def __init__(self, model, order: list | None = None):
        super().__init__()
        self.model = model
        self.order = order or []

    @property
    def _wrapped_model(self):
        return self.model

    def __getitem__(self, key):
        try:
            return collections.UserDict.__getitem__(self, key)
        except KeyError:
            collections.UserDict.__setitem__(self, key, copy.deepcopy(self.model))
            return self[key]


class RegressorParallel(BaseParallel, base.MultiTargetRegressor):
    """A multi-output model that creates independent regressors for each target.

    This will create one model per output. The prediction of the first output will be used as a
    feature in the second output. The prediction for the second output will be used as a feature
    for the third, etc. This "chain model" is therefore capable of capturing dependencies between
    outputs.

    Parameters
    ----------
    model
        The regression model used to make predictions for each target.
    order
        A list with the targets order in which to construct the chain. If `None` then the order
        will be inferred from the order of the keys in the target.
    """

    def __init__(self, model: base.Regressor):
        super().__init__(model)

    @classmethod
    def _unit_test_params(cls):
        yield {"model": linear_model.LinearRegression()}

    def learn_one(self, x, y, **kwargs):
        for o in y:
            result = {}
            if isinstance(y[o], dict):
                for key, value in y[o].items():
                    result[key] = value
            else:
                result[0] = y[o]
            self[o].learn_one(x, result, **kwargs)

        return self

    def predict_one(self, x, **kwargs):
        y_pred = {}
        for o, reg in self.items():
            y_pred[o] = reg.predict_one(x, **kwargs)

        y_pred_res = {}
        if isinstance(y_pred[0], dict):
            for inner_dict in y_pred.values():
                y_pred_res.update(inner_dict)
        else:
            y_pred_res = y_pred

        return y_pred_res
    
class RegressorFullParallel(BaseParallel, base.MultiTargetRegressor):
    """A multi-output model that creates independent regressors for each target.

    This will create one model per output. The prediction of the first output will be used as a
    feature in the second output. The prediction for the second output will be used as a feature
    for the third, etc. This "chain model" is therefore capable of capturing dependencies between
    outputs.

    Parameters
    ----------
    model
        The regression model used to make predictions for each target.
    order
        A list with the targets order in which to construct the chain. If `None` then the order
        will be inferred from the order of the keys in the target.
    """

    def __init__(self, model: base.Regressor):
        super().__init__(model)

    @classmethod
    def _unit_test_params(cls):
        yield {"model": linear_model.LinearRegression()}

    def learn_one(self, x, y, **kwargs):
        for o in y:
            result = {}
            if isinstance(y[o], dict):
                for key, value in y[o].items():
                    result[key] = value
            else:
                result[0] = y[o]
            self[o].learn_one(x[o], result, **kwargs)

        return self

    def predict_one(self, x, **kwargs):
        y_pred = {}
        for o, reg in self.items():
            y_pred[o] = reg.predict_one(x[o], **kwargs)

        y_pred_res = {}
        if isinstance(y_pred[list(y_pred.keys())[0]], dict):
            for inner_dict in y_pred.values():
                y_pred_res.update(inner_dict)
        else:
            y_pred_res = y_pred

        return y_pred_res