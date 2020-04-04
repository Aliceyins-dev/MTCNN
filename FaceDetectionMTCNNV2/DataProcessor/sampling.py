from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
from PIL import Image


class FaceDataset(Dataset):

    mean = torch.tensor([0.5327, 0.4363, 0.3878]).reshape(3, 1, 1)
    std = torch.tensor([0.3018, 0.2817, 0.2800]).reshape(3, 1, 1)

    def __init__(self, path):
        self.path = path
        self.datasets = []

        self.datasets.extend(open(os.path.join(self.path, "positive.txt")).readlines())
        self.datasets.extend(open(os.path.join(self.path, "negative.txt")).readlines())
        self.datasets.extend(open(os.path.join(self.path, "part.txt")).readlines())

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        strs = self.datasets[index].strip().split(" ")
        img_path = os.path.join(self.path, strs[0])
        classify = torch.tensor([int(strs[1])], dtype=torch.float32)
        offset = torch.tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])], dtype=torch.float32)
        img_data = torch.tensor(np.array(Image.open(img_path)), dtype=torch.float32).permute(2, 0, 1) / 255.
        img_data = (img_data - FaceDataset.mean) / FaceDataset.std
        return img_data, classify, offset


if __name__ == '__main__':
    dataset = FaceDataset(r"E:\FaceDetectionMTCNNV2\datasets\48")
    print(len(dataset))
    print(dataset[0])
    # data = DataLoader(dataset=dataset, batch_size=40001, shuffle=True)
    # data = next(iter(data))[0]
    # print(data)
    # mean = torch.mean(data, dim=(0, 2, 3))
    # std = torch.std(data, dim=(0, 2, 3))
    # print(mean)
    # print(std)



"""
12:
tensor([0.5330, 0.4360, 0.3879])
tensor([0.2985, 0.2813, 0.2797])
24:
tensor([0.5328, 0.4370, 0.3888])
tensor([0.3027, 0.2826, 0.2810])
48:
tensor([0.5324, 0.4358, 0.3868])
tensor([0.3042, 0.2813, 0.2792])

tensor([0.5327, 0.4363, 0.3878])
tensor([0.3018, 0.2817, 0.2800])

"""




