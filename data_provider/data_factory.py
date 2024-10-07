from data_provider.data_loader import Dataset_flame
from torch.utils.data import DataLoader

data_dict = {
    'FLAME': Dataset_flame
}


def data_provider(args, flag):
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    data_set = Dataset_flame(args, flag)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
    
