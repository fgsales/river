from __future__ import annotations

import csv
import datetime as dt
import random

from .. import base
from . import utils

from collections import deque
import itertools

__all__ = ["iter_csv", "iter_csv_with_moving_window"]


class DictReader(csv.DictReader):
    """Overlay on top of `csv.DictReader` which allows sampling."""

    def __init__(self, fraction, rng, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fraction = fraction
        self.rng = rng

    def __next__(self):
        if self.line_num == 0:
            self.fieldnames

        row = next(self.reader)

        if self.fraction < 1:
            while self.rng.random() > self.fraction:
                row = next(self.reader)

        return dict(zip(self.fieldnames, row))


def iter_csv(
    filepath_or_buffer,
    target: str | list[str] | None = None,
    converters: dict | None = None,
    parse_dates: dict | None = None,
    drop: list[str] | None = None,
    drop_nones=False,
    fraction=1.0,
    compression="infer",
    seed: int | None = None,
    field_size_limit: int | None = None,
    **kwargs,
) -> base.typing.Stream:
    """Iterates over rows from a CSV file.

    Reading CSV files can be quite slow. If, for whatever reason, you're going to loop through
    the same file multiple times, then we recommend that you to use the `stream.Cache` utility.

    Parameters
    ----------
    filepath_or_buffer
        Either a string indicating the location of a file, or a buffer object that has a
        `read` method.
    target
        A single target column is assumed if a string is passed. A multiple output scenario
        is assumed if a list of strings is passed. A `None` value will be assigned to each `y`
        if this parameter is omitted.
    converters
        All values in the CSV are interpreted as strings by default. You can use this parameter to
        cast values to the desired type. This should be a `dict` mapping feature names to callables
        used to parse their associated values. Note that a callable may be a type, such as `float`
        and `int`.
    parse_dates
        A `dict` mapping feature names to a format passed to the `datetime.datetime.strptime`
        method.
    drop
        Fields to ignore.
    drop_nones
        Whether or not to drop fields where the value is a `None`.
    fraction
        Sampling fraction.
    compression
        For on-the-fly decompression of on-disk data. If this is set to 'infer' and
        `filepath_or_buffer` is a path, then the decompression method is inferred for the
        following extensions: '.gz', '.zip'.
    seed
        If specified, the sampling will be deterministic.
    field_size_limit
        If not `None`, this will be passed to the `csv.field_size_limit` function.
    kwargs
        All other keyword arguments are passed to the underlying `csv.DictReader`.

    Returns
    -------

    By default each feature value will be of type `str`. You can use the `converters` and
    `parse_dates` parameters to convert them as you see fit.

    Examples
    --------

    Although this function is designed to handle different kinds of inputs, the most common
    use case is to read a file on the disk. We'll first create a little CSV file to illustrate.

    >>> tv_shows = '''name,year,rating
    ... Planet Earth II,2016,9.5
    ... Planet Earth,2006,9.4
    ... Band of Brothers,2001,9.4
    ... Breaking Bad,2008,9.4
    ... Chernobyl,2019,9.4
    ... '''
    >>> with open('tv_shows.csv', mode='w') as f:
    ...     _ = f.write(tv_shows)

    We can now go through the rows one by one. We can use the `converters` parameter to cast
    the `rating` field value as a `float`. We can also convert the `year` to a `datetime` via
    the `parse_dates` parameter.

    >>> from river import stream

    >>> params = {
    ...     'converters': {'rating': float},
    ...     'parse_dates': {'year': '%Y'}
    ... }
    >>> for x, y in stream.iter_csv('tv_shows.csv', **params):
    ...     print(x, y)
    {'name': 'Planet Earth II', 'year': datetime.datetime(2016, 1, 1, 0, 0), 'rating': 9.5} None
    {'name': 'Planet Earth', 'year': datetime.datetime(2006, 1, 1, 0, 0), 'rating': 9.4} None
    {'name': 'Band of Brothers', 'year': datetime.datetime(2001, 1, 1, 0, 0), 'rating': 9.4} None
    {'name': 'Breaking Bad', 'year': datetime.datetime(2008, 1, 1, 0, 0), 'rating': 9.4} None
    {'name': 'Chernobyl', 'year': datetime.datetime(2019, 1, 1, 0, 0), 'rating': 9.4} None

    The value of `y` is always `None` because we haven't provided a value for the `target`
    parameter. Here is an example where a `target` is provided:

    >>> dataset = stream.iter_csv('tv_shows.csv', target='rating', **params)
    >>> for x, y in dataset:
    ...     print(x, y)
    {'name': 'Planet Earth II', 'year': datetime.datetime(2016, 1, 1, 0, 0)} 9.5
    {'name': 'Planet Earth', 'year': datetime.datetime(2006, 1, 1, 0, 0)} 9.4
    {'name': 'Band of Brothers', 'year': datetime.datetime(2001, 1, 1, 0, 0)} 9.4
    {'name': 'Breaking Bad', 'year': datetime.datetime(2008, 1, 1, 0, 0)} 9.4
    {'name': 'Chernobyl', 'year': datetime.datetime(2019, 1, 1, 0, 0)} 9.4

    Finally, let's delete the example file.

    >>> import os; os.remove('tv_shows.csv')

    """

    # Set the field size limit
    limit = csv.field_size_limit()
    if field_size_limit is not None:
        csv.field_size_limit(field_size_limit)

    # If a file is not opened, then we open it
    buffer = filepath_or_buffer
    if not hasattr(buffer, "read"):
        buffer = utils.open_filepath(buffer, compression)

    for x in DictReader(fraction=fraction, rng=random.Random(seed), f=buffer, **kwargs):
        if drop:
            for i in drop:
                del x[i]

        # Cast the values to the given types
        if converters is not None:
            for i, t in converters.items():
                x[i] = t(x[i])

        # Drop Nones
        if drop_nones:
            for i in list(x):
                if x[i] is None:
                    del x[i]

        # Parse the dates
        if parse_dates is not None:
            for i, fmt in parse_dates.items():
                x[i] = dt.datetime.strptime(x[i], fmt)

        # Separate the target from the features
        y = None
        if isinstance(target, list):
            y = {name: x.pop(name) for name in target}
        elif target is not None:
            y = x.pop(target)

        yield x, y

    # Close the file if we opened it
    if buffer is not filepath_or_buffer:
        buffer.close()

    # Reset the file size limit to it's original value
    csv.field_size_limit(limit)


def fillna(history_row, last_valid_values, nan_values=[0]):

    filled_history_row = history_row.copy()

    for key, value in history_row.items():
        if value in nan_values and key in last_valid_values:
            filled_history_row[key] = last_valid_values[key]
        elif value not in nan_values: # If the value is valid, update the valid values
            last_valid_values[key] = value
    
    return filled_history_row, last_valid_values


def sliding_window_iter_csv(
    filepath_or_buffer,
    target: str | list[str] | None = None,
    converters: dict | None = None,
    parse_dates: dict | None = None,
    drop: list[str] | None = None,
    drop_nones=False,
    ffill=False,
    fraction=1.0,
    compression="infer",
    seed: int | None = None,
    field_size_limit: int | None = None,
    past_history: int = 5,
    forecast_horizon: int = 2,
    **kwargs,
) -> base.typing.Stream:
    """ Iterates over rows from a CSV file with a sliding window for features and targets.

    Additional Parameters
    ---------------------
    past_history : int
        Number of past records to include in the sliding window.
    forecast_horizon : int
        Number of future records to predict.

    Other parameters are the same as in iter_csv.
    """

    # Set the field size limit
    limit = csv.field_size_limit()
    if field_size_limit is not None:
        csv.field_size_limit(field_size_limit)

    # Open the file or buffer
    buffer = filepath_or_buffer
    if not hasattr(buffer, "read"):
        buffer = utils.open_filepath(buffer, compression)

    reader = DictReader(fraction=fraction, rng=random.Random(seed), f=buffer, **kwargs)

    # Initialize history and future buffers
    history_buffer = deque(maxlen=past_history+forecast_horizon)
    # future_buffer = deque(maxlen=forecast_horizon)

    last_valid_values = {}

    for row in reader:
        history_buffer.append(row.copy())
        
        if len(history_buffer) < past_history + forecast_horizon:
            continue

        window_data = {}
        # iterate until past history for past and the forecast horizon for future
        for i, history_row in enumerate(history_buffer):

            if ffill: # History buffer forward filling
                history_row, last_valid_values = fillna(history_row, last_valid_values)

            offset = i - past_history + 1
            # key = ""
            # if i < past_history:
            #     key = f'{key}_{offset}'
            # else:
            #     key = f'{key}_+{offset}'
                
            for key, value in history_row.items():
                    window_data[f'{key}_{offset}'] = value

        if drop:
            window_data = {k: v for k, v in window_data.items() if k.split('_')[0] not in drop}

        # Apply converters
        if converters is not None:
            for key, conv in converters.items():
                # Apply converter to each related window_data key
                for window_key in list(window_data.keys()):
                    if window_key.startswith(key):
                        window_data[window_key] = conv(window_data[window_key])
        else:
            # Convert to float if no converters are specified
            for window_key in list(window_data.keys()):
                window_data[window_key] = float(window_data[window_key])

        # Drop Nones
        if drop_nones:
            window_data = {k: v for k, v in window_data.items() if v is not None}


        # Parse dates
        if parse_dates is not None:
            for key, fmt in parse_dates.items():
                # Apply date parsing to each related window_data key
                for window_key in list(window_data.keys()):
                    if window_key.startswith(key):
                        window_data[window_key] = dt.datetime.strptime(window_data[window_key], fmt)

        # Separate the target from the features
        y = {}
        if target is None:
            # If target is None, select all variables with future offsets as targets
            for key in list(window_data.keys()):
                if key.endswith(tuple([f"_{i}" for i in range(1, forecast_horizon + 1)])):
                    y[key] = window_data.pop(key)
        else:
            target_keys = [target] if isinstance(target, str) else target
            for target_key in target_keys:
                # Extract future values as targets
                for i in range(1, forecast_horizon + 1):
                    y[f'{target_key}_+{i}'] = window_data.pop(f'{target_key}_+{i}', None)

        yield window_data, y

    # Close the file if we opened it
    if buffer is not filepath_or_buffer:
        buffer.close()

    # Reset the field size limit to its original value
    csv.field_size_limit(limit)
