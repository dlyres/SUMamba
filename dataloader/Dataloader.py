import os
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


class SSVEPDataset(Dataset):
    def __init__(self, dataset_name, cross_validation, train=None):
        self.train = train
        self.samples = []
        self.dataset_name = dataset_name
        if cross_validation:
            self.root_dir = os.path.join('./dataset/', self.dataset_name + '/cross_validation')
        if self.train == True:
            self.root_dir = os.path.join('./dataset/', self.dataset_name + '/train')
        if self.train == False:
            self.root_dir = os.path.join('./dataset/', self.dataset_name + '/test')
        for label_folder in os.listdir(self.root_dir):
            label_path = os.path.join(self.root_dir, label_folder)
            for mat_file in os.listdir(label_path):
                if mat_file.endswith('.mat'):
                    mat_data = loadmat(os.path.join(label_path, mat_file))
                    self.samples.append({'data': mat_data, 'label': int(label_folder) - 1})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self.samples[index]['data']
        label = self.samples[index]['label']
        if data['sample_frequence'] is not None:
            # print(type(data['sample_frequence']))
            data = torch.Tensor(data['sample_frequence'])
            return data, label


def make_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



