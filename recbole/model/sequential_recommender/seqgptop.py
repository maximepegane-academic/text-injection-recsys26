r"""
GPTopFreq
################################################

"""

import torch

from recbole.model.sequential_recommender import SeqPersonalPop


class SeqGPTop(SeqPersonalPop):
    r"""SeqGPTop also called GP-Topfreq in litterature is a fundamental model that always recommend a user's most purchased
    item, followed by the most popular items"""

    def __init__(self, config, dataset):
        super(SeqGPTop, self).__init__(config, dataset)
        self.pop = torch.tensor(dataset.inter_matrix().sum(axis=0)).squeeze()
        self.pop = self.pop / self.pop.sum()
        self.pop = self.pop.to(self.device)

    def calculate_loss(self, interaction):
        return self.fake_loss

    def predict(self, interaction):
        result = (
            (interaction[self.ITEM_SEQ] == interaction[self.ITEM_ID]).sum(dim=1).float()
        )
        result += self.pop[interaction[self.ITEM_ID]]
        return result

    def full_sort_predict(self, interaction):
        history = interaction[self.ITEM_SEQ]
        batch_size = interaction.length
        result = torch.zeros(
            (batch_size, self.n_items), device=self.device, dtype=torch.float
        )
        result = result.scatter_add(dim=1, src=(history > 0).float(), index=history)
        result += self.pop

        return result
