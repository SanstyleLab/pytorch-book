from pathlib import Path
import torch


class BaseModel:
    def __init__(self, name, is_train, checkpoints_dir, gpu_ids):
        self.gpu_ids = gpu_ids
        self.is_train = is_train

        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.name = name
        self.save_dir = Path(checkpoints_dir)/self.name
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        # initialize optimizers
        self.schedulers = []
        self.optimizers = []

    def set_save_path(self, network_label, epoch_label):
        save_filename = f"{epoch_label}_{network_label}.h5"
        save_path = self.save_dir/save_filename
        return save_path.as_posix()

    def save_network(self, network, network_label, epoch_label):
        save_path = self.set_save_path(network_label, epoch_label)
        # 保存模型
        torch.save(network.cpu().state_dict(), save_path)
        if len(self.gpu_ids) and torch.cuda.is_available():
            network.cuda(self.gpu_ids[0])

    def load_network(self, network, network_label, epoch_label):
        save_path = self.set_save_path(network_label, epoch_label)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        if self.optimizers:
            lr = self.optimizers[0].param_groups[0]['lr']
            print(f'learning rate = {lr:.7g}')

    def set_input(self, inputs):
        self.input = inputs

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def forward(self):
        NotImplemented

    def test(self):
        NotImplemented

    def get_image_paths(self):
        NotImplemented

    def optimize_parameters(self):
        NotImplemented

    def save(self, label):
        NotImplemented
