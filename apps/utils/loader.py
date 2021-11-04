from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None  # 加载大文件
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 跳过损坏的文件


class Record:
    def __init__(self, bunch):
        self.bunch = bunch
        self.class_names = list(self.bunch.keys())
        self.class_dict = {class_name: k
                           for k, class_name in enumerate(self.class_names)}

    @property
    def records(self):
        _record = []
        for class_name, names in self.bunch.items():
            for name in names:
                _record.append([name, self.class_dict[class_name]])
        return _record

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        name, label = self.records[idx]
        # 如果存在 RGBA，则转换为 RGB
        # print(name)
        image = Image.open(name).convert('RGB')
        return image, label


if __name__ == '__main__':
    from tools.toml import load_option
    from opt.dataset import init_dataset

    init_dataset(200)
    bunch = load_option('../result/dataset/all.toml')
