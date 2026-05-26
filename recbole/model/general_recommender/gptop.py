r"""
GPTopFreq
################################################

"""

import torch

from recbole.model.general_recommender.personalpop import PersonalPop


class GPTop(PersonalPop):
    r"""GPTop also called GP-Topfreq in litterature is a fundamental model that always recommend a user's most purchased
    item, followed by the most popular items"""

    def __init__(self, config, dataset):
        super(GPTop, self).__init__(config, dataset)
        self.total_purchase_count = torch.zeros(
            1, dtype=torch.float64, device=self.device, requires_grad=False
        )
        self.item_purchase_count = torch.zeros(
            self.n_items, device=self.device, requires_grad=False
        )

    def calculate_loss(self, interaction):
        items, users = interaction[self.ITEM_ID], interaction[self.USER_ID]
        self.user_purchase_per_item[items, users] += 1
        self.user_purchase_count[users] += 1
        self.item_purchase_count[items] += 1
        self.total_purchase_count += interaction.length

        return self.fake_loss

    def predict(self, interaction):
        users, items = interaction[self.USER_ID], interaction[self.ITEM_ID]
        result = self.user_purchase_per_item[items, users].view(-1)
        global_result = self.item_purchase_count[items].div(self.total_purchase_count)
        result[torch.logical_not(result)] += global_result[torch.logical_not(result)]
        result /= self.user_purchase_count[users]
        return result

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        result = self.user_purchase_per_item[:, users].view(-1)
        global_result = self.item_purchase_count.div(self.total_purchase_count)
        result[torch.logical_not(result)] += global_result[torch.logical_not(result)]
        result /= self.user_purchase_count[users]
        return result.view(-1)
