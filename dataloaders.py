#%% 
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils_top import load_tensors
from utils.utils_midi import load_npy_files, convert_list_to_data

sys.path.append("C:/Users/lucas/Documents/GitHub/RNN_gradient_problem/raw_data")
main_path = "C:/Users/lucas/Documents/GitHub/RNN_gradient_problem/"


class top_dataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.from_numpy(self.sequences[idx]).float()  # Sequence has size [T, 6]
        label = torch.from_numpy(self.labels[idx]).float()  # Label has size [2] each value being interpreted has the probability of being 'A'
        return sequence, label  


def get_top_dataloaders(batch_size, path=main_path+"raw_data/top", T=50):
    X_train, y_train, X_eval, X_test, y_eval, y_test = load_tensors(path, T)
    train_dataset = top_dataset(X_train, y_train)
    eval_dataset = top_dataset(X_eval, y_eval)
    test_dataset = top_dataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True)

    return train_dataloader, eval_dataloader, test_dataloader


class midi_dataset(Dataset):
    def __init__(self, folder_path) -> None:
        X_list = load_npy_files(folder_path)
        self.X, self.y = convert_list_to_data(X_list)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        return X, y


def get_midi_dataloaders(batch_size, folder_path = "raw_data/midi/npy"):
    train_dataset = midi_dataset(folder_path + "/train")
    eval_dataset = midi_dataset(folder_path + "/eval")
    test_dataset = midi_dataset(folder_path + "/test")

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True)

    return train_dataloader, eval_dataloader, test_dataloader