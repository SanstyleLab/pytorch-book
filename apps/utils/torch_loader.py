from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.loader import Record
from tools.mask import mask_iter
from tools.toml import load_option
from opt.dataset import init_dataset


class CustomDataset(Dataset, Record):
    def __init__(self, bunch, transform=None, target_transform=None):
        super().__init__(bunch)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        name, label = self.records[idx]
        # 如果存在 RGBA，则转换为 RGB
        image = Image.open(name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class Loader:
    def __init__(self, fine_size, batch_size, mask, dataset_opt='../result/dataset/all.toml'):
        self.mask_root = mask
        self.bunch = load_option(dataset_opt)
        self.fine_size = fine_size
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(1024, scale=(0.5, 1.0)),
            transforms.Resize((self.fine_size, self.fine_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self._dataset = CustomDataset(
            self.bunch, transform=self.transform)

    @property
    def dataset(self):
        _loader = DataLoader(self._dataset,
                             batch_size=self.batch_size,
                             shuffle=True)
        return _loader

    @property
    def maskset(self):
        return mask_iter(self.mask_root, self.fine_size)
