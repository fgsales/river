from __future__ import annotations

from copy import deepcopy
import random
import statistics

from river import base, tree, drift

from .nodes.branch import DTBranch
from .nodes.isouptr_nodes import LeafAdaptiveMultiTarget, LeafMeanMultiTarget, LeafModelMultiTarget
from .isoup_tree_regressor import iSOUPTreeRegressor
from .split_criterion import IntraClusterVarianceReductionSplitCriterion
from .splitter import Splitter


class iSOUPAdaptiveTreeRegressor(iSOUPTreeRegressor):
    """Incremental Structured Output Prediction Tree (iSOUP-Tree) for multi-target regression.
    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    delta
        Allowed error in split decision, a value closer to 0 takes longer to
        decide.
    tau
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mean' - Target mean</br>
        - 'model' - Uses the model defined in `leaf_model`</br>
        - 'adaptive' - Chooses between 'mean' and 'model' dynamically</br>
    leaf_model
        The regression model(s) used to provide responses if `leaf_prediction='model'`. It can
        be either a regressor (in which case it is going to be replicated to all the targets)
        or a dictionary whose keys are target identifiers, and the values are instances of
        `river.base.Regressor.` If not provided, instances of `river.linear_model.LinearRegression`
        with the default hyperparameters are used for all the targets. If a dictionary is passed
        and not all target models are specified, copies from the first model match in the
        dictionary will be used to the remaining targets.
    model_selector_decay
        The exponential decaying factor applied to the learning models' squared errors, that
        are monitored if `leaf_prediction='adaptive'`. Must be between `0` and `1`. The closer
        to `1`, the more importance is going to be given to past observations. On the other hand,
        if its value approaches `0`, the recent observed errors are going to have more influence
        on the final decision.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric attributes
        should be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.TEBSTSplitter` is used if `splitter` is `None`.
    min_samples_split
        The minimum number of samples every branch resulting from a split candidate must have
        to be considered valid.
    binary_split
        If True, only allow binary splits.
    max_size
        The max size of the tree, in Megabytes (MB).
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.

    References
    ----------
    [^1]: Aljaž Osojnik, Panče Panov, and Sašo Džeroski. "Tree-based methods for online
        multi-target regression." Journal of Intelligent Information Systems 50.2 (2018): 315-339.

    Examples
    --------

    >>> import numbers
    >>> from river import compose
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from river import tree

    >>> dataset = datasets.SolarFlare()

    >>> num = compose.SelectType(numbers.Number) | preprocessing.MinMaxScaler()
    >>> cat = compose.SelectType(str) | preprocessing.OneHotEncoder()

    >>> model = tree.iSOUPTreeRegressor(
    ...     grace_period=100,
    ...     leaf_prediction='model',
    ...     leaf_model={
    ...         'c-class-flares': linear_model.LinearRegression(l2=0.02),
    ...         'm-class-flares': linear_model.PARegressor(),
    ...         'x-class-flares': linear_model.LinearRegression(l2=0.1)
    ...     }
    ... )

    >>> pipeline = (num + cat) | model
    >>> metric = metrics.multioutput.MicroAverage(metrics.MAE())

    >>> evaluate.progressive_val_score(dataset, pipeline, metric)
    MicroAverage(MAE): 0.426177

    """

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int | None = None,
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "adaptive",
        leaf_model: base.Regressor | dict | None = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        min_samples_split: int = 5,
        bootstrap_sampling: bool = True,
        drift_window_threshold: int = 300,
        drift_detector: base.DriftDetector | None = None,
        switch_significance: float = 0.05,
        binary_split: bool = False,
        max_size: float = 500.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            delta=delta,
            tau=tau,
            leaf_prediction=leaf_prediction,
            leaf_model=leaf_model,
            model_selector_decay=model_selector_decay,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            min_samples_split=min_samples_split,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self.bootstrap_sampling = bootstrap_sampling
        self.drift_window_threshold = drift_window_threshold
        self.drift_detector = drift_detector if drift_detector is not None else drift.ADWIN()
        self.switch_significance = switch_significance
        self.seed = seed
        self.split_criterion: str = "icvr"  # intra cluster variance reduction
        self.targets: set = set()

        self._n_alternate_trees = 0
        self._n_pruned_alternate_trees = 0
        self._n_switch_alternate_trees = 0
        self._norm_dist = statistics.NormalDist()

        self._rng = random.Random(self.seed)

    @property
    def _mutable_attributes(self):
        return {"grace_period", "delta", "tau", "drift_window_threshold", "switch_significance"}

    @property
    def n_alternate_trees(self):
        return self._n_alternate_trees

    @property
    def n_pruned_alternate_trees(self):
        return self._n_pruned_alternate_trees

    @property
    def n_switch_alternate_trees(self):
        return self._n_switch_alternate_trees

    @property
    def summary(self):
        summ = super().summary
        summ.update(
            {
                "n_alternate_trees": self.n_alternate_trees,
                "n_pruned_alternate_trees": self.n_pruned_alternate_trees,
                "n_switch_alternate_trees": self.n_switch_alternate_trees,
            }
        )
        return summ



    def learn_one(self, x, y, *, sample_weight: float = 1.0) -> iSOUPAdaptiveTreeRegressor:  # type: ignore
        """Incrementally train the model with one sample.

        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for
          the instance and update the leaf node statistics.
        * If growth is allowed and the number of instances that the leaf has
          observed between split attempts exceed the grace period then attempt
          to split.

        Parameters
        ----------
        x
            Instance attributes.
        y
            Target values.
        sample_weight
            The weight of the passed sample.
        """
        # Update target set
        self.targets.update(y.keys())

        super().learn_one(x, y, sample_weight=sample_weight)  # type: ignore

        return self

    def predict_one(self, x):
        pred = {}
        if self._root is not None:
            if isinstance(self._root, DTBranch):
                leaf = self._root.traverse(x, until_leaf=True)
            else:
                leaf = self._root

            pred = leaf.prediction(x, tree=self)  # type: ignore
        return pred
