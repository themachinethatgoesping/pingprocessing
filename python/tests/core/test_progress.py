# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0


from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator
from tqdm.auto import tqdm

class MyCustomProgressClass:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable
        self.kwargs = kwargs

    def __iter__(self):
        return iter(self.iterable)

    def __len__(self):
        return len(self.iterable)


class TestProgress:
    """
    This class contains unit tests for the get_progress_iterator function.
    """

    def test_get_progress_iterator_without_progress(self):
        # Given
        iterable = [1, 2, 3]

        # When
        it = get_progress_iterator(iterable, False)

        # Then
        assert list(it) == iterable
        assert it.__class__ == list

        for i,r in enumerate(it):
            assert r == iterable[i]

    def test_get_progress_iterator_with_progress(self):
        # Given
        iterable = [1, 2, 3]

        # When
        it = get_progress_iterator(iterable, True)

        # Then
        assert list(it) == iterable
        assert it.__class__ == tqdm

        for i,r in enumerate(it):
            assert r == iterable[i]


    def test_get_progress_iterator_explicit_tqdm(self):
        # Given
        iterable = [1, 2, 3]

        # When
        it = get_progress_iterator(iterable, tqdm)

        # Then
        assert list(it) == iterable
        assert it.__class__ == tqdm

        for i,r in enumerate(it):
            assert r == iterable[i]

    def test_get_progress_iterator_with_custom_progress_class(self):
        # Given
        iterable = [1, 2, 3]
        progress_class = MyCustomProgressClass

        # When
        it = get_progress_iterator(iterable, progress_class)

        # Then
        assert list(it) == iterable
        assert it.__class__ == MyCustomProgressClass

        for i,r in enumerate(it):
            assert r == iterable[i]
