from Nets.mtcnn_net import ONet
import os
from Trainers import trainer

if __name__ == '__main__':
    onet = ONet()
    if not os.path.exists("../models"):
        os.makedirs("../models")
    train = trainer.Trainer(onet,  "../models/o_net.pth", r"E:\FaceDetectionMTCNNV2\datasets\48")
    train.train(0.0001, alpha=0.4)