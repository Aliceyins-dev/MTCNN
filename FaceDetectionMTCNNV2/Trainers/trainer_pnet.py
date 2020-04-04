from Nets.mtcnn_net import PNet
import os
from Trainers import trainer

if __name__ == '__main__':
    pnet = PNet()
    if not os.path.exists("../models"):
        os.makedirs("../models")
    train = trainer.Trainer(pnet, '../models/p_net.pth', r"E:\FaceDetectionMTCNNV2\datasets\12")
    train.train(0.01, alpha=0.9)
