r"""
TIFU-KNN
################################################

"""

import torch
import torch.nn as nn
from tqdm import tqdm
from recbole.model.abstract_recommender import SequentialRecommender

from recbole.utils import InputType, ModelType, unique_with_index
from typing import Union


def soft_round(
    x: torch.Tensor, alpha: Union[float, torch.Tensor] = 5, eps: float = 1e-7
) -> torch.Tensor:
    """
    Differentiable approximation to the round function in a single function.

    Args:
        x (torch.Tensor): The input tensor to be rounded.
        alpha (Union[float, torch.Tensor]): Controls the smoothness of the
            approximation. Larger values result in a harder, more step-like
            rounding. Can be a scalar or a tensor.
        eps (float): A small constant for numerical stability.

    Returns:
        torch.Tensor: The soft-rounded tensor, with the same shape as x.
    """
    # Ensure alpha is a tensor for consistency
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)

    alpha_bounded = torch.clamp(alpha, min=eps)

    m = torch.floor(x) + 0.5
    r = x - m
    z = torch.tanh(alpha_bounded / 2.0) * 2.0
    y = m + torch.tanh(alpha_bounded * r) / z

    return torch.where(alpha < eps, x, y)


class PositiveSoftInteger(nn.Module):
    def forward(self, X):
        X = nn.functional.softplus(X)
        X = soft_round(X)
        return X


class FNNRec(SequentialRecommender):
    r"""TIFU-KNN is a model published in Hu et al. 2020 (https://doi.org/10.48550/arXiv.2006.00556)
    It uses a twice time weighted aggregation of a user's history, to represent a user, then uses this representation and
    the  K Nearest Neighbor's representation to make a prediction about a user's next purchase
    """

    input_type = InputType.POINTWISE
    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(FNNRec, self).__init__(config, dataset)
        self.number_fake_neighbors = config["number_fake_neighbors"]
        self.number_nearest_neighbors = config["number_nearest_neighbors"]
        self.alpha = config["alpha"]
        self.seed = config["seed"]
        self.fake_neighbor = torch.nn.Embedding(
            self.number_fake_neighbors, self.n_items
        )
        self.pop = dataset.item_popularity.to(self.device)
        torch.nn.utils.parametrizations.parametrize.register_parametrization(
            self.fake_neighbor, "weight", PositiveSoftInteger()
        )
        self.loss = torch.nn.BCELoss()

    def find_nearest_neigbor(self, input):
        nearest_neighbors_idx = self.torh.cdist(input, self.fake_neighbor.weight).topk(
            self.number_nearest_neighbors, dim=1
        )
        return nearest_neighbors_idx

    def forward(self, history):
        batch_size = history.shape[0]
        out = torch.zeros((batch_size, self.n_items), device=self.device)
        out.scatter_add_(dim=1, src=(history > 0).float(), index=history)
        dists, indexes = torch.cdist(out, self.fake_neighbor.weight).topk(
            self.number_nearest_neighbors, dim=1, largest=False
        )

        dist_score = torch.nn.functional.softmax(-dists, dim=1)
        selected_neighbors = self.fake_neighbor(indexes)

        average_neighbor = (selected_neighbors * dist_score.unsqueeze(-1)).sum(dim=1)
        return average_neighbor

    def calculate_loss(self, interaction):
        pos_items = interaction[self.POS_ITEM_ID]
        history = interaction[self.ITEM_SEQ]
        batch_size = interaction.length

        labels = torch.zeros([batch_size, self.n_items], device=self.device)
        labels.scatter_add_(dim=1, src=(pos_items > 0).float(), index=pos_items)
        # explore_pos_item
        out = self.forward(history)
        out = torch.nn.functional.softmax(out, dim=1)
        loss = self.loss(out, labels)
        loss -= 10 * self.loss(
            out,
            (nn.functional.softmax(self.pop).unsqueeze(0)).repeat_interleave(
                batch_size, dim=0
            ),
        )
        loss += nn.functional.mse_loss(out.sum(dim=1), (history > 0).sum(dim=1).float())
        return loss

    def predict(self, interaction):
        return self.forward(interaction[self.ITEM_SEQ])

    def full_sort_predict(self, interaction):
        return self.forward(interaction[self.ITEM_SEQ])
