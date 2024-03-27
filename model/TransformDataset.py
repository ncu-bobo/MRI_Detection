from torch.utils.data import Dataset


class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

    def __len__(self):
        return len(self.dataset)
