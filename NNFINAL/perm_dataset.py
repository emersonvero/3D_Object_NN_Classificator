import os
import gzip
import torch
from torch.utils.data import Dataset
from itertools import permutations

class Permuted1Dataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)
        self.permutation = [1, 0, 2]  # Define the desired permutation

    def __len__(self):
        return len(self.file_list) * 2  # Double the dataset size

    def __getitem__(self, index):
        file_index = index // 2
        file_name = self.file_list[file_index]
        file_path = os.path.join(self.folder_path, file_name)
        with gzip.open(file_path, 'rb') as f:
            loaded_data = torch.load(f, map_location=torch.device('cpu'))

        data = loaded_data[0]
        label = loaded_data[1]

        # Perform the permutation on the data tensor
        data = data.permute(*self.permutation)

        # Perform any necessary preprocessing on the data and label
        # ...

        return data, label


class Permuted2Dataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)
        self.permutations = [[1, 0, 2], [0, 2, 1]]  # Define the desired permutations

    def __len__(self):
        return len(self.file_list) * len(self.permutations)  # Triple the dataset size

    def __getitem__(self, index):
        file_index = index // len(self.permutations)
        permutation_index = index % len(self.permutations)
        file_name = self.file_list[file_index]
        file_path = os.path.join(self.folder_path, file_name)
        with gzip.open(file_path, 'rb') as f:
            loaded_data = torch.load(f, map_location=torch.device('cpu'))

        data = loaded_data[0]
        label = loaded_data[1]

        # Perform the permutation on the data tensor
        permutation = self.permutations[permutation_index]
        data = data.permute(*permutation)

        # Perform any necessary preprocessing on the data and label
        # ...

        return data, label


class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)
        self.permutations = [[1, 0, 2], [0, 2, 1], [2, 1, 0]]  # Define the desired permutations

    def __len__(self):
        return len(self.file_list) * len(self.permutations)  # Quadruple the dataset size

    def __getitem__(self, index):
        file_index = index // len(self.permutations)
        permutation_index = index % len(self.permutations)
        file_name = self.file_list[file_index]
        file_path = os.path.join(self.folder_path, file_name)
        with gzip.open(file_path, 'rb') as f:
            loaded_data = torch.load(f, map_location=torch.device('cpu'))

        data = loaded_data[0]
        label = loaded_data[1]

        # Perform the permutation on the data tensor
        permutation = self.permutations[permutation_index]
        data = data.permute(*permutation)

        # Perform any necessary preprocessing on the data and label
        # ...

        return data, label

import os
import gzip
import torch
from torch.utils.data import Dataset
from itertools import permutations

class PermDataset(Dataset):
    def __init__(self, folder_path, num_permutations):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)
        self.permutations = list(permutations(range(3)))[:num_permutations]  # Generate all possible permutations
        self.dataset_size = len(self.file_list) * len(self.permutations)  # Calculate the dataset size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        file_index = index // len(self.permutations)
        permutation_index = index % len(self.permutations)
        file_name = self.file_list[file_index]
        file_path = os.path.join(self.folder_path, file_name)
        with gzip.open(file_path, 'rb') as f:
            loaded_data = torch.load(f, map_location=torch.device('cpu'))

        data = loaded_data[0]
        label = loaded_data[1]

        # Perform the permutation on the data tensor
        permutation = self.permutations[permutation_index]
        data = data.permute(*permutation)

        # Perform any necessary preprocessing on the data and label
        # ...

        return data, label

from torch.utils.data import Dataset
from itertools import permutations

class PermLabeledDataset(Dataset):
    def __init__(self, folder_path, num_permutations):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)
        self.permutations = list(permutations(range(3)))[:num_permutations]  # Generate all possible permutations
        self.dataset_size = len(self.file_list) * len(self.permutations)  # Calculate the dataset size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        file_index = index // len(self.permutations)
        permutation_index = index % len(self.permutations)
        file_name = self.file_list[file_index]
        file_path = os.path.join(self.folder_path, file_name)
        with gzip.open(file_path, 'rb') as f:
            loaded_data = torch.load(f, map_location=torch.device('cpu'))

        data = loaded_data[0]
        label = loaded_data[1]

        # Perform the permutation on the data tensor
        permutation = self.permutations[permutation_index]
        data = data.permute(*permutation)

        # Update the label to include the permutation information
        label = torch.cat((label, torch.tensor(permutation_index).unsqueeze(0)))

        # Perform any necessary preprocessing on the data and label
        # ...

        return data, label



