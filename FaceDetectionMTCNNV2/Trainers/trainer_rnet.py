from Nets.mtcnn_net import RNet
import os
from Trainers import trainer


if __name__ == '__main__':
    rnet = RNet()
    if not os.path.exists("../models"):
        os.makedirs("../models")
    train = trainer.Trainer(rnet, "../models/r_net.pth", r"E:\FaceDetectionMTCNNV2\datasets\24")
    train.train(0.001, alpha=0.6)
