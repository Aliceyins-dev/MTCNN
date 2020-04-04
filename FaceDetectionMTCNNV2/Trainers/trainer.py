import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from DataProcessor.sampling import FaceDataset
import torch.optim.lr_scheduler as lr_scheduler


class Trainer:
    def __init__(self, net, param_path, dataset_path):
        self.net = net
        self.param_path = param_path
        self.dataset_path = dataset_path

        self.cls_lossfunc = nn.BCELoss()
        self.offset_lossfunc = nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.scheduler = lr_scheduler.StepLR(self.optim, 20, gamma=0.8)

        if os.path.exists(self.param_path):
            net.load_state_dict(torch.load(self.param_path))
        else:
            print("No Param")

    def train(self, stopvalue, alpha):
        face_dataset = FaceDataset(self.dataset_path)
        loader_data = DataLoader(dataset=face_dataset, batch_size=512, shuffle=True, num_workers=4)
        losses = []
        loss = 0
        epoch = 0
        while True:
            for i, (img_data, classify, offset) in enumerate(loader_data):
                out_classify, out_offset = self.net(img_data)
                out_classify = out_classify.reshape(-1, 1)
                out_offset = out_offset.reshape(-1, 4)
                # print(classify.shape)
                # print(out_classify.shape)

                classify_mask = torch.lt(classify, 2)
                # print(classify_mask.shape)
                cls_orig = torch.masked_select(classify, classify_mask)
                cls_prob = torch.masked_select(out_classify, classify_mask)
                # print(cls_orig.shape)
                # print(cls_prob.shape)

                classify_loss = self.cls_lossfunc(cls_prob, cls_orig)

                offset_mask = torch.gt(classify, 0)
                offset_orig = torch.masked_select(offset, offset_mask)
                offset_prob = torch.masked_select(out_offset, offset_mask)
                offset_loss = self.offset_lossfunc(offset_prob, offset_orig)

                loss = alpha * classify_loss.float() + (1 - alpha) * offset_loss.float()

                if i % 10 == 0:
                    losses.append(loss)
                    print("loss：{0}, cls_loss:{1}, offset_loss:{2}".format(loss, classify_loss, offset_loss))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            torch.save(self.net.state_dict(), self.param_path)
            self.scheduler.step(epoch)
            epoch += 1
            print("save success:{}".format(epoch))
            if loss < stopvalue:
                break







