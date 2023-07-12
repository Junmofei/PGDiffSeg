from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np



def load_train(args):
    # get ids 
    with open(f"{args.root}{args.data_path}/{args.train_list}.txt", "r") as f:
        train_list = f.read()   #读取文本
    train_list = train_list.split('\n')

    
    train_set = DataGen(args.root, args.data_path, train_list)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)

    return train_loader

def load_test(args):
    # get ids 
    with open(f"{args.root}{args.data_path}/{args.test_list}.txt", "r") as f: 
        test_list = f.read()   #读取文本
    test_list = test_list.split('\n')

    test_set = TestGen(args.root, args.data_path, test_list)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    return DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)

class DataGen(Dataset):
    def __init__(self, root, data_path, name_list):
        self.root = root
        self.data_path = data_path
        self.name_list = name_list

    def __getitem__(self, index):
        data = np.load(f'{self.root}{self.data_path}/{self.name_list[index]}', allow_pickle=True)
        return data[0], data[1]  # npy = [image, mask, classes]
    
    def __len__(self):
        return len(self.name_list)
    

class TestGen(Dataset):
    def __init__(self, root, data_path, name_list):
        self.root = root
        self.data_path = data_path
        self.name_list = name_list

    def __getitem__(self, index):
        data = np.load(f'{self.root}{self.data_path}/{self.name_list[index]}', allow_pickle=True)
        return self.name_list[index], data[0] , data[1]
    
    def __len__(self):
        return len(self.name_list)