import pytest
import torch
from recbole.model.sequential_recommender import TIFUKNN
from unittest.mock import Mock


@pytest.fixture
def mock_config():
    return {
        "number_of_group": 2,
        "within_group_decay": 0.9,
        "global_decay": 0.8,
        "number_neighbors": 1,
        "knn_method": "full",
        "precompute_neighbors": False,
        "distance": "euclidean",
        "alpha": 0.7,
        "seed": 42,
        "knn_batch_size": 256,
        "USER_ID_FIELD": "shopper_id",
        "ITEM_ID_FIELD": "product_id",
        "TIME_FIELD": "transaction_date",
        "LIST_SUFFIX": "_list",
        "MAX_ITEM_LIST_LENGTH": 100,
        "NEG_PREFIX": "neg_",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "device": "cpu",
    }


@pytest.fixture
def mock_dataset():
    dataset = Mock()
    dataset.purchase_counts = torch.zeros(5, dtype=torch.long)  # 5 users
    dataset.n_users = 5
    dataset.n_items = 10
    dataset.num = lambda x: dataset.n_users if x == "shopper_id" else dataset.n_items
    return dataset


def test_full_sort_tifu_knn(mock_config, mock_dataset):
    """Test prediction for completely new users with no history"""
    model = TIFUKNN(mock_config, mock_dataset)
    model.device = mock_config["device"]
    model.unseen_user_mask = torch.ones(5, dtype=torch.bool, device=model.device)
    model.unseen_user_mask[0:3] = ~model.unseen_user_mask[0:3]
    # Create interaction with 2 unseen users
    interaction = {
        model.USER_ID: torch.tensor([3, 4], dtype=torch.int64, device=model.device),
        model.ITEM_SEQ: torch.tensor(
            [[1, 2, 3, 1, 1, 3], [3, 4, 5, 3, 3, 4]],
            dtype=torch.int64,
            device=model.device,
        ),
        model.TIME_FIELD
        + "_list": torch.tensor(
            [[1.0, 2.0, 2.0, 3.0, 4.0, 4.0], [1.0, 1.0, 1.0, 2.0, 3.0, 3.0]],
            dtype=torch.float32,
            device=model.device,
        ),
        model.ITEM_SEQ_LEN: torch.tensor(
            [6, 6], dtype=torch.int64, device=model.device
        ),
    }

    result = model.full_sort_predict(interaction)

    # Verify basic output properties
    assert result.shape == (2, 10)
    assert torch.isfinite(result).all()

    # Verify user 3's item weights
    user3 = result[0]
    assert user3[1] > user3[2]  # Time decay check (item 1 purchased earlier)
    assert user3[3:].sum() == 0  # Only items 1 and 2 should have weights
