r"""
TIFU-KNN
################################################

"""

import torch
from tqdm import tqdm
from recbole.model.abstract_recommender import SequentialRecommender

from recbole.utils import InputType, ModelType


class TIFUKNN(SequentialRecommender):
    r"""TIFU-KNN is a model published in Hu et al. 2020 (https://doi.org/10.48550/arXiv.2006.00556)
    It uses a twice time weighted aggregation of a user's history, to represent a user, then uses this representation and
    the  K Nearest Neighbor's representation to make a prediction about a user's next purchase
    """

    input_type = InputType.POINTWISE
    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(TIFUKNN, self).__init__(config, dataset)
        self.number_of_group = config["number_of_group"]
        self.within_group_decay = config["within_group_decay"]
        self.global_decay = config["global_decay"]
        self.number_neighbors = config["number_neighbors"]
        self.knn_method = config["knn_method"]
        self.precompute_neighbors = config["precompute_neighbors"]
        self.distance_name = config["distance"]
        self.distance = self.init_distance()
        self.alpha = config["alpha"]
        self.seed = config["seed"]
        self.knn_batch_size = config["knn_batch_size"]
        self.weighted_history_per_user = torch.zeros(
            (self.n_users, self.n_items), device=self.device, requires_grad=False
        )
        self.user_purchase_weighted_count = torch.zeros(
            self.n_users, dtype=torch.float64, device=self.device, requires_grad=False
        )

        self.item_purchase_weighted_count = torch.zeros(
            self.n_items, device=self.device, requires_grad=False
        )
        self.precomputed_neighbors = torch.zeros(
            (self.n_users, self.n_items),
            dtype=torch.float64,
            device=self.device,
            requires_grad=False,
        )
        self.purchase_counts = dataset.purchase_counts
        self.total_purchase_count = torch.zeros(
            1, dtype=torch.float64, device=self.device, requires_grad=False
        )
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.querying_index = None
        self.unseen_user_mask = torch.ones(
            self.n_users, dtype=torch.bool, device=self.device, requires_grad=False
        )

    def init_distance(self):
        return lambda x1, x2: torch.cdist(x1, x2, p=2)
        # planned to try other distance but batch calculation work mostly for Minkowski type norms (p-norms)

    def find_nearest_neigbor(self, input):
        if self.knn_method == "full":
            _, nearest_neighbors_idx = self.distance(
                input,
                self.weighted_history_per_user[
                    torch.logical_not(self.unseen_user_mask)
                ],
            ).topk(self.number_neighbors, dim=1)
            return nearest_neighbors_idx
        if (
            self.knn_method == "approximate"
        ):  # saves some time for 10k sample you go from 15mn to 6-7 mn for CPU inference
            nearest_neighbors_idx, _ = self.index.query(
                input.cpu(), k=self.number_neighbors + 1
            )
            return nearest_neighbors_idx

    def build_ANN_index(self):
        if self.knn_method == "approximate":
            from pynndescent import NNDescent

            self.index = NNDescent(
                self.weighted_history_per_user.cpu(),
                metric=self.distance_name,
                random_state=self.seed,
            )

    def neighbors_precomputing(self):
        if self.precompute_neighbors:
            precomputed_neighbors = self.find_nearest_neigbor(
                self.weighted_history_per_user
            )
            batch_size = self.knn_batch_size
            load_bar = tqdm(
                range(
                    0,
                    self.n_users,
                    batch_size,
                ),
                desc="Precomputing neighbors",
                ncols=100,
                colour="green",
            )
            for i in load_bar:
                batch_neighbors_idx = precomputed_neighbors[i : i + batch_size]
                batch_neighbors_representation = self.weighted_history_per_user[
                    batch_neighbors_idx
                ]
                batch_neighbors_representation = (
                    (
                        batch_neighbors_representation
                        / self.user_purchase_weighted_count[
                            batch_neighbors_idx
                        ].unsqueeze(2)
                    )
                    .nan_to_num(0)
                    .mean(dim=1)
                )
                self.precomputed_neighbors[i : i + batch_size] = (
                    batch_neighbors_representation
                )

    def check_unseen_user(self):
        self.unseen_user_mask = self.weighted_history_per_user.sum(dim=1) == 0

    def forward(self):
        pass

    def calculate_loss(self, interaction):

        (
            users,
            items,
        ) = (
            interaction[self.USER_ID],
            interaction[self.ITEM_ID],
        )
        num_basket_per_user, purchase_rank_order = (
            self.purchase_counts[interaction[self.USER_ID] - 1],
            interaction[self.TIME_FIELD],
        )

        # Number of basket to go in each partition, is different for each person
        group_size_estimation = num_basket_per_user // self.number_of_group
        remainder_baskets = num_basket_per_user % self.number_of_group

        # the paper stipulates that if a number of basket is unevenly divided by the number of group then the first group
        # will contain up to n_group-1 extra basket, and the other will have n_remaining_basket / n_group

        # check if we are in the first bucket range (potentially with extra basket)

        is_in_extra_basket_range = (
            purchase_rank_order <= group_size_estimation + remainder_baskets
        )

        # index for the other buckets
        non_extra_bucket_index = (
            (purchase_rank_order - remainder_baskets) * self.number_of_group
        ) // (num_basket_per_user - remainder_baskets)
        bucket_index = torch.where(
            is_in_extra_basket_range, 0, non_extra_bucket_index
        )  # if True  # then  # else

        bucket_index = bucket_index.clamp(min=0, max=self.number_of_group - 1)  # safety
        group_size = (
            group_size_estimation + is_in_extra_basket_range * remainder_baskets
        )

        # once we have the bucket size, the global index, and the bucket index for each basket
        # we don't really need to actually partition the baskets to weigh them, we can directly calculate the within
        # group index and infer the final weighting from that

        within_group_order = purchase_rank_order % group_size
        weighted_value = purchase_rank_order * self.within_group_decay ** (
            group_size - within_group_order
        )
        weighted_value = weighted_value * self.global_decay ** (
            self.number_of_group - 1 - bucket_index
        )
        weighted_value = weighted_value.nan_to_num(nan=0.0)

        self.weighted_history_per_user[users, items] += weighted_value
        self.item_purchase_weighted_count[items] += weighted_value
        self.user_purchase_weighted_count[users] += weighted_value
        self.total_purchase_count += interaction.length
        return self.fake_loss

    def predict(self, interaction):
        users, items = interaction[self.USER_ID], interaction[self.ITEM_ID]

        unique_users, count = users.unique(return_counts=True)

        # to possibly reduce the amount of computations for neighbor search we only get unique users and repeat in the output
        users_representation = self.weighted_history_per_user[unique_users]

        # getting neighbors vector
        neighbor_idx = self.find_nearest_neigbor(users_representation)
        neighbor_representation = self.weighted_history_per_user[neighbor_idx]
        neighbor_partial_representation = (
            neighbor_representation[:, items]
            / self.user_purchase_weighted_count[neighbor_idx].unsqueeze(2)
        ).mean(dim=1)

        users_partial_representation = users_representation[
            :, items
        ] / self.user_purchase_weighted_count[unique_users].unsqueeze(1)
        # weighting the two vectors according to alpha
        result = (
            self.alpha * users_partial_representation
            + (1 - self.alpha) * neighbor_partial_representation
        )
        result = torch.repeat_interleave(result, count, dim=0)
        return result.squeeze(-1)

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        unique_users, unique_index, reverse_index_users = unique_with_index(users)
        n_unique = unique_users.shape[0]

        # Initialize representations using matrix operations
        user_repr = torch.zeros((n_unique, self.n_items), device=self.device)
        neighbor_repr = torch.zeros_like(user_repr)

        # Process seen users
        seen_mask = ~self.unseen_user_mask[unique_users]
        if seen_mask.any():
            seen_users = unique_users[seen_mask]
            user_repr[seen_mask] = self.weighted_history_per_user[seen_users]

            # Vectorized neighbor processing
            neighbor_idx = self.find_nearest_neigbor(
                self.weighted_history_per_user[seen_users]
            )
            neighbor_repr[seen_mask] = (
                self.weighted_history_per_user[neighbor_idx].nan_to_num(0).mean(dim=1)
            )

        # Process unseen users
        if torch.any(~seen_mask):
            unseen_idx = torch.where(~seen_mask)[0]
            batch_idx = unique_index[self.unseen_user_mask[unique_users]]

            item_hist = interaction[self.ITEM_ID + "_list"][batch_idx]
            trans_times = interaction[self.TIME_FIELD + "_list"][batch_idx]

            # Vectorized temporal calculations
            min_time = trans_times.min(dim=1).values
            reordered_trans = trans_times - min_time.unsqueeze(1)
            num_basket = trans_times.max(dim=1).values - min_time + 1

            # Optimized grouping logic
            m = self.number_of_group
            group_size, remainder = num_basket // m, num_basket % m
            in_extra_group = reordered_trans <= (group_size + remainder).unsqueeze(1)

            # Vectorized decay calculations
            bucket_idx = torch.where(
                in_extra_group,
                0,
                ((reordered_trans - remainder.unsqueeze(1)) * m)
                / (num_basket - remainder).unsqueeze(1),
            ).long()

            valid_group_size = group_size.unsqueeze(
                1
            ) + in_extra_group * remainder.unsqueeze(1)
            within_decay = self.within_group_decay ** (
                valid_group_size - (reordered_trans % valid_group_size)
            )
            global_decay = self.global_decay ** (m - 1 - bucket_idx)

            weighted_value = (reordered_trans * within_decay * global_decay).nan_to_num(
                0
            )

            # Efficient sparse update using scatter_add
            user_repr.view(-1).scatter_add_(
                0,
                (unseen_idx.view(-1, 1) * self.n_items + item_hist).view(-1),
                weighted_value.view(-1),
            )

            # Neighbor calculation with existing users only
            neighbor_idx = self.find_nearest_neigbor(user_repr[~seen_mask])
            neighbor_repr[~seen_mask] = (
                self.weighted_history_per_user[neighbor_idx].nan_to_num(0).mean(dim=1)
            )

        user_repr = user_repr.div(user_repr.sum(dim=1, keepdim=True)).nan_to_num(0)
        neighbor_repr = (
            neighbor_repr / neighbor_repr.sum(dim=1, keepdim=True)
        ).nan_to_num(0)
        # Stabilized combination
        combined = self.alpha * user_repr + (1 - self.alpha) * neighbor_repr
        combined_sum = combined.sum(dim=1, keepdim=True).clamp_min(1e-10)
        return (combined / combined_sum)[reverse_index_users]


def unique_with_index(x):
    unique, inverse, count = torch.unique(
        x, sorted=True, return_inverse=True, return_counts=True
    )
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inversed, perm = inverse.flip([0]), perm.flip([0])
    perm = inversed.new_empty(unique.size(0)).scatter_(0, inversed, perm)
    return unique, perm, inverse
