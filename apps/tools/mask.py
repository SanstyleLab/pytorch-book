import numpy as np
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class Maskset(Dataset):
    def __init__(self, mask_root, transform, shuffle=True):
        super().__init__()
        self.transform = transform
        root = Path(mask_root)
        self.paths = np.array([path.as_posix() for path in root.iterdir()])
        if shuffle:
            np.random.shuffle(self.paths)

    def _mask(self, path):
        with Image.open(path) as mask:
            mask = self.transform(mask.convert('RGB'))
        return mask

    def __getitem__(self, index):
        paths = self.paths[index]
        if isinstance(index, slice):
            n_mask = len(self)
            # masks = (self._mask(path) for path in paths)
            masks = [self._mask(path) for path in paths]
        else:
            masks = self._mask(paths)
        return masks

    def __len__(self):
        return len(self.paths)


def mask_iter(mask_root, fine_size):
    transform = transforms.Compose([transforms.Resize((fine_size, fine_size)),
                                    transforms.ToTensor()])
    return Maskset(mask_root, transform)
