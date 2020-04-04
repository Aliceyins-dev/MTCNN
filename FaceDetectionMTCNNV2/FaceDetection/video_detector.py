from Nets.mtcnn_net import PNet, RNet, ONet
import torch
from torchvision.transforms import transforms
import time
import numpy as np
from Tools.utils import nms, convert_to_square
import cv2
from PIL import Image, ImageDraw, ImageFont


class Detector:
    def __init__(self, pnet_param="../models/p_net.pth", rnet_param="../models/r_net.pth", onet_param="../models/o_net.pth"):
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5327, 0.4363, 0.3878), (0.3018, 0.2817, 0.2800))
        ])

    def face_detect(self, image):
        pstart_time = time.time()
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        pend_time = time.time()
        p_time = pend_time - pstart_time

        rstart_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        rend_time = time.time()
        r_time = rend_time - rstart_time

        ostart_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        oend_time = time.time()
        o_time = oend_time - ostart_time

        time_sum = p_time + r_time + o_time
        print("totle time:{0}, p_time:{1}, r_time:{2}, o_time:{3}".format(time_sum, p_time, r_time, o_time))
        # print("totle time:{}".format(time_sum))
        return onet_boxes

    def __pnet_detect(self, image):
        img_w, img_h = image.size
        min_side_len = min(img_w, img_h)
        scale = 1
        boxes = []
        while min_side_len > 12:
            img_data = self.__img_transforms(image)
            img_data = img_data.unsqueeze(0)

            _classify, _offset = self.pnet(img_data)
            classify, offset = _classify[0][0], _offset[0]
            indexes = torch.nonzero(torch.gt(classify, 0.6))
            for ids in indexes:
                boxes.append(self.__box(ids, offset, classify[ids[0], ids[1]], scale))
            scale *= 0.709
            _weight = int(img_w * scale)
            _height = int(img_h * scale)
            image = image.resize((_weight, _height))
            min_side_len = np.minimum(_weight, _height)
        return nms(np.stack(boxes), 0.3, False)

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):

        _x1 = int(start_index[1] * stride) / scale
        _y1 = int(start_index[0] * stride) / scale
        _x2 = int(start_index[1] * stride + side_len) / scale
        _y2 = int(start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]

    def __rnet_detect(self, image, pnet_boxes):
        """
        R网络检测过程
        :param image:
        :param pnet_boxes:
        :return:
        """
        img_dataset = []
        pnet_boxes = convert_to_square(pnet_boxes)
        for box in pnet_boxes:
            crop_x1 = int(box[0])
            crop_y1 = int(box[1])
            crop_x2 = int(box[2])
            cropy_y2 = int(box[3])

            img = image.crop((crop_x1, crop_y1, crop_x2, cropy_y2))
            img = img.resize((24, 24))
            imgdata = self.__img_transforms(img)
            img_dataset.append(imgdata)
        img_dataset = torch.stack(img_dataset)
        classify, offset = self.rnet(img_dataset)
        classify, offset = classify.numpy(), offset.numpy()

        boxes = []
        indexes, _ = np.where(classify > 0.7)
        for idx in indexes:
            _box = pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            ow = _x2 - _x1
            oh = _y2 - _y1
            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = classify[idx][0]
            boxes.append([x1, y1, x2, y2, cls])

        return nms(np.array(boxes), 0.3)

    def __onet_detect(self, image, rnet_boxes):
        img_dataset = []
        rnet_boxes = convert_to_square(rnet_boxes)
        for box in rnet_boxes:
            crop_x1 = int(box[0])
            crop_y1 = int(box[1])
            crop_x2 = int(box[2])
            crop_y2 = int(box[3])

            img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            img = img.resize((48, 48))
            img_data = self.__img_transforms(img)
            img_dataset.append(img_data)
        img_dataset = torch.stack(img_dataset)
        classify, offset = self.onet(img_dataset)
        classify, offset = classify.numpy(), offset.numpy()
        boxes = []
        indexes, _ = np.where(classify > 0.9)
        for idx in indexes:
            _box = rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            ow = _x2 - _x1
            oh = _y2 - _y1
            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = classify[idx][0]
            boxes.append([x1, y1, x2, y2, cls])
        return nms(np.array(boxes), 0.3, isMin=True)


def resize_image(img, scale):
    height, width, channel = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    return img_resized


if __name__ == '__main__':
    with torch.no_grad() as grad:
        detector = Detector()
        cap = cv2.VideoCapture("../videos/video1.mp4")
        while True:
            isSuccess, frame = cap.read()
            if isSuccess:
                try:
                    frame = resize_image(frame, 0.25)
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    start_time = time.time()
                    bboxes = detector.face_detect(image)
                    draw = ImageDraw.Draw(image)
                    # font = ImageFont.truetype('utils/simkai.ttf', 30)
                    # FPS = 1.0 / (time.time() - start_time)
                    # draw.text((10, 10), 'FPS: {:.1f}'.format(FPS), fill=(0, 0, 0), font=font)

                    for box in bboxes:
                        x1 = int(box[0])
                        y1 = int(box[1])
                        x2 = int(box[2])
                        y2 = int(box[3])
                        draw.rectangle((x1, y1, x2, y2), outline='red', width=3)
                    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                except:
                    print('detect error')
                cv2.imshow('video', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
