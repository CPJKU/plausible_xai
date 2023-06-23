from torch.utils.data import Subset


class ImageNetSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.samples = [dataset.samples[i] for i in indices]


class AdversarialImagenetSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.file_list = [dataset.file_list[i] for i in indices]
