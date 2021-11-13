from pathlib import Path
import numpy as np


class Tree:
    def __init__(self, root) -> None:
        '''注意，root 下面不能存在子目录'''
        self.root = Path(root)
        self.special_suffix = ['.DNG', '.CR2']

    @property
    def full_names(self):
        '''全部文件名称'''
        return {name.as_posix()
                for name in self.root.iterdir()}

    @property
    def suffix(self):
        '''全部文件后缀'''
        return {n.suffix for n in self.names}

    @property
    def special_names(self):
        '''全部文件名称'''
        return {name.as_posix()
                for name in self.root.iterdir()
                if name.suffix in self.special_suffix}

    @property
    def names(self):
        '''全部可用图片名称'''
        return self.full_names - self.special_names

    def __len__(self):
        return len(self.names)

    def take_shuffle(self, num):
        '''随机选择 num 个图片'''
        _names = list(self.names)
        m = len(_names)
        index = np.arange(m)
        np.random.shuffle(index)
        return np.take(_names, index[:num]).tolist()


class PathSet:
    def __init__(self, root) -> None:
        self.root = root

    @property
    def parent_paths(self):
        # 获取全部数据的路径
        _paths = [path.as_posix()
                  for path in Path(self.root).iterdir()]
        return _paths

    @property
    def bunch(self):
        _bunch = {Path(name).name: Tree(name).names
                  for name in self.parent_paths}
        return _bunch

    @property
    def names(self):
        _names = set()
        for val in self.bunch.values():
            _names = _names.union(val)
        return _names
