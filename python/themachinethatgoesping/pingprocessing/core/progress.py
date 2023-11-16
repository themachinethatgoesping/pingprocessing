# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

from tqdm.auto import tqdm

from typing import Any, Callable, Iterable, Optional, Type, Union

def get_progress_iterator(
    iteratable: Iterable[Any], 
    progress: Optional[Union[bool, Type[Callable[..., Any]]]] = False, 
    **kwargs: Any) -> Iterable[Any]:
    """
    Returns an iterator that optionally displays a progress bar.

    Args:
        iteratable: The iterable to iterate over.
        progress: If True, displays a progress bar using the default tqdm implementation.
                  If a callable is provided, it should be a progress bar implementation that takes an iterable as its first argument.
                  If False, returns the original iterable without any progress bar.

    Returns:
        An iterator that optionally displays a progress bar.
    """
    if progress is not False:
        if progress is True:
            progress = tqdm

        return progress(iteratable, delay = 2, **kwargs)
    else:
        return iteratable