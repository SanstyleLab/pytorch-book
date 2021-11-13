from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.loader_all import Record
from tools.mask import mask_iter


class CustomDataset(Dataset, Record):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        name = self.names[idx]
        # 如果存在 RGBA，则转换为 RGB
        image = Image.open(name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


class Loader:
    def __init__(self, fine_size, batch_size, mask, root):
        self.root = root
        self.mask_root = mask
        self.fine_size = fine_size
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(1024, scale=(0.5, 1.0)),
            transforms.Resize((self.fine_size, self.fine_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def trainset(self):
        self._trainset = CustomDataset(self.root, transform=self.transform)
        _loader = DataLoader(self._trainset,
                             batch_size=self.batch_size,
                             shuffle=True)
        return _loader

    def maskset(self):
        return mask_iter(self.mask_root, self.fine_size)


class LoaderTest:
    def __init__(self, fine_size, batch_size, mask, root):
        self.root = root
        self.mask_root = mask
        self.fine_size = fine_size
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((self.fine_size, self.fine_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def valset(self):
        self._trainset = CustomDataset(self.root, transform=self.transform)
        _loader = DataLoader(self._trainset,
                             batch_size=self.batch_size,
                             shuffle=True)
        return _loader

    def maskset(self):
        return mask_iter(self.mask_root, self.fine_size)