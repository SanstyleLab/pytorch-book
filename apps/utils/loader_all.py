from PIL import Image, ImageFile

from tools.split import PathSet

Image.MAX_IMAGE_PIXELS = None  # 加载大文件
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 跳过损坏的文件


class Record:
    def __init__(self, root):
        self.names = list(PathSet(root).names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        # 如果存在 RGBA，则转换为 RGB
        # print(name)
        image = Image.open(name).convert('RGB')
        return image, name