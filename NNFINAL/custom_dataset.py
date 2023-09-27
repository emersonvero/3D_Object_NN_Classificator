import os
import gzip
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        file_path = os.path.join(self.folder_path, file_name)
        with gzip.open(file_path, 'rb') as f:
            loaded_data = torch.load(f, map_location=torch.device('cpu'))

        data = loaded_data[0]
        label = loaded_data[1]

        # Perform any necessary preprocessing on the data and label
        # ...

        return data, label
