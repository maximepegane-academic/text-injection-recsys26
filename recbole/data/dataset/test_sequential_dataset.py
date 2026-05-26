import pytest
import numpy as np
from recbole.data.dataset.sequential_dataset import SequentialDataset

case_1 = (
    np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),  # uids
    np.array([1, 1, 1, 2, 2, 3, 1, 1, 2, 2, 2]),  # timestamps
    50,  # array_length
    [
        slice(0, 0),
        slice(0, 0),
        slice(0, 0),  # user 1, 3 interactions, 0 interaction in previous basket
        slice(0, 3),
        slice(0, 3),  # user 1, 2 interactions, 3 interaction in previous basket
        slice(0, 5),  # user 1, 1 interactions, 5 interaction in previous baskets
        slice(6, 6),
        slice(6, 6),  # user 2, 2 interactions, 0 interaction in previous basket
        slice(6, 8),
        slice(6, 8),
        slice(6, 8),
    ],  # user 2, 3 interactions, 2 interaction in previous basket
)

case_2 = (
    np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),  # uids
    np.array([1, 1, 1, 2, 2, 3, 1, 1, 2, 2, 2]),  # timestamps
    1,  # array_length
    [
        slice(0, 0),
        slice(0, 0),
        slice(0, 0),  # user 1, 3 interactions, 0 interaction in previous basket
        slice(2, 3),
        slice(2, 3),  # user 1, 2 interactions, 1 interaction in previous basket
        slice(4, 5),  # user 1, 1 interactions, 1 interaction in previous baskets
        slice(6, 6),
        slice(6, 6),  # user 2, 2 interactions, 0 interaction in previous basket
        slice(7, 8),
        slice(7, 8),
        slice(7, 8),
    ],  # user 2, 3 interactions, 1 interaction in previous basket
)


@pytest.mark.parametrize(
    "uid_array, time_array, max_sequence_length, expected", [case_1, case_2]
)
def test_build_item_id_lists(uid_array, time_array, max_sequence_length, expected):
    slicing, _, _ = SequentialDataset.build_item_id_lists(
        uid_array, time_array, max_sequence_length
    )
    assert (slicing == np.array(expected)).all()
