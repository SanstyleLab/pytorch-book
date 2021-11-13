from pathlib import Path

from tools.split import Tree
from tools.toml import write_option, load_option
from tools.file import mkdir


def get_paths(bunch):
    # 获取全部数据的路径
    paths = [path.as_posix()
             for path in Path(bunch.root).iterdir()
             if path.name not in  ['生成用', '周边风景照片']]
    return paths


def get_info(paths):
    # 查看：每个类别的图片个数
    return {name: len(Tree(name)) for name in paths}


def init_dataset(num=200):
    # 载入原始数据的配置：
    origin = load_option('../origin/all.toml')
    paths = get_paths(origin)
    print(get_info(paths))

    dataset_bunch = {Path(name).name: Tree(
        name).take_shuffle(num) for name in paths}

    opt_root = Path('../result/dataset')
    mkdir(opt_root)
    opt_all = opt_root/'all.toml'

    # 初始化数据配置
    write_option(dataset_bunch, opt_all)
    origin.pop('root')
    return origin


if __name__ == '__main__':
    init_dataset()
