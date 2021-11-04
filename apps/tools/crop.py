from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tools.toml import load_option
from tools.mask import mask_iter

Image.MAX_IMAGE_PIXELS = None # 加载大文件
ImageFile.LOAD_TRUNCATED_IMAGES = True # 跳过损坏的文件


class BunchPath:
    def __init__(self, class_path) -> None:
        self.path = Path(class_path)

    @property
    def root(self):
        return self.path.name

    @property
    def names(self):
        return [_p.name for _p in self.path.iterdir()]

    def full_name(self, name):
        return (self.path/name).as_posix()

    def __len__(self):
        '''文件个数'''
        return len(self.names)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            L = [self.full_name(name)
                 for name in self.names[idx]]
            return L
        else:
            return self.full_name(self.names[idx])


class BiBunch:
    '''两层的 BunchPath'''

    def __init__(self, root) -> None:
        self.class_bunch = BunchPath(root)

    def other_images(self):
        _names = []
        for class_path in self.class_bunch:
            bunch = BunchPath(class_path)
            for name in bunch:
                if name.endswith('.CR2') or name.endswith('.DNG'):
                    _names.append((name, bunch.root))
                else:
                    continue
        return _names

    def dataset(self):
        _names = []
        for class_path in self.class_bunch:
            bunch = BunchPath(class_path)
            for name in bunch:
                if name.endswith('.CR2') or name.endswith('.DNG'):
                    continue
                else:
                    _names.append((name, bunch.root))
        return _names

# def random_crop(image, crop_shape, padding=None):
#     oshape = image.size

#     if padding:
#         oshape_pad = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
#         img_pad = Image.new("RGB", (oshape_pad[0], oshape_pad[1]))
#         img_pad.paste(image, (padding, padding))
        
#         nh = random.randint(0, oshape_pad[0] - crop_shape[0])
#         nw = random.randint(0, oshape_pad[1] - crop_shape[1])
#         image_crop = img_pad.crop((nh, nw, nh+crop_shape[0], nw+crop_shape[1]))

#         return image_crop
#     else:
#         print("WARNING!!! nothing to do!!!")
#         return image

class PILDataset:
    def __init__(self, root):
        self.bunch = BiBunch(root)
        self.dataset = self.bunch.dataset()
        self.class_names = self.bunch.class_bunch.names
        self.class_dict = {name: k for k, name in enumerate(self.class_names)}

    def __len__(self):
        '''图片个数'''
        return len(self.dataset)

    def __getitem__(self, idx):
        name, label = self.dataset[idx]
        # 如果存在 RGBA，则转换为 RGB
        # print(name)
        image = Image.open(name).convert('RGB')
        label = self.class_dict[label]
        return image, label


class CustomDataset(Dataset, PILDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        name, label = self.dataset[idx]
        # 如果存在 RGBA，则转换为 RGB
        image = Image.open(name).convert('RGB')
        label = self.class_dict[label]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class Loader:
    def __init__(self, opt_path='options/loader-custom.toml'):
        self.opt = load_option(opt_path)
        self.fine_size = self.opt['fine_size']
        self.batch_size = self.opt['batch_size']
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(1024,scale=(0.5,1.0)),
            transforms.Resize((self.fine_size, self.fine_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self._trainset = CustomDataset(
            self.opt['root'], transform=self.transform)

    @property
    def trainset(self):
        _loader = DataLoader(self._trainset,
                             batch_size=self.batch_size,
                             shuffle=True)
        return _loader

    @property
    def maskset(self):
        return mask_iter(self.opt['mask_root'], self.fine_size)
