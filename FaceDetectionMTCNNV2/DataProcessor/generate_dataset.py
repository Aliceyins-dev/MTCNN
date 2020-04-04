import os
from PIL import Image
import numpy as np
import traceback
from Tools import utils


anno_src = r"E:\datasets\list_bbox_celeba_ALL.txt"
image_src = r"E:\datasets\img_celeba_ALL"
save_path = r"E:\FaceDetectionMTCNNV2\datasets"

float_num = [0.1, 0.2, 0.5, 0.65, 0.75, 0.8, 0.9, 0.95, 0.96, 0.99]


def generate_sample(facesize, stopvalue):
    """
    生成样本数据集
    :param facesize: 样本尺寸
    :param stopvalue: 停止条件
    :return:
    """
    print("the size:{}".format(facesize))

    positive_dir = os.path.join(save_path, str(facesize), "positive")
    negative_dir = os.path.join(save_path, str(facesize), "negative")
    part_dir = os.path.join(save_path, str(facesize), "part")

    for dir_path in [positive_dir, negative_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    positive_anno_filename = os.path.join(save_path, str(facesize), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(facesize), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(facesize), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0
    try:
        positive_anno_file = open(positive_anno_filename, mode="w")
        negative_anno_file = open(negative_anno_filename, mode="w")
        part_anno_file = open(part_anno_filename, mode="w")

        for i, line in enumerate(open(anno_src)):
            if i < 2:
                continue
            try:
                strs = line.split()
                img_name = strs[0].strip()
                print("image name:{}".format(img_name))
                img_file = os.path.join(image_src, img_name)

                with Image.open(img_file) as img:
                    img_w, img_h = img.size

                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = x1 + w
                    y2 = y1 + h

                    if max(img_w, img_h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue
                    boxes = [[x1, y1, x2, y2]]

                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    side_len = max(w, h)

                    seed = float_num[np.random.randint(0, len(float_num))]

                    for i in range(5):
                        side_len_off = side_len + np.random.randint(int(-side_len * seed), int(side_len * seed))
                        cx_off = cx + np.random.randint(int(-cx * seed), int(cx * seed))
                        cy_off = cy + np.random.randint(int(-cy * seed), int(cy * seed))

                        x1_off = cx_off - side_len_off / 2
                        y1_off = cy_off - side_len_off / 2
                        x2_off = x1_off + side_len_off
                        y2_off = y1_off + side_len_off

                        if x1_off < 0 or y1_off < 0 or x2_off > img_w or y2_off > img_h:
                            continue

                        offset_x1 = (x1 - x1_off) / side_len_off
                        offset_y1 = (y1 - y1_off) / side_len_off
                        offset_x2 = (x2 - x2_off) / side_len_off
                        offset_y2 = (y2 - y2_off) / side_len_off

                        crop_box = [x1_off, y1_off, x2_off, y2_off]
                        crop_img = img.crop(crop_box)
                        face_resize = crop_img.resize((facesize, facesize))

                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        if iou > 0.68:
                            positive_anno_file.write("positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(positive_count, 1, offset_x1, offset_y1, offset_x2, offset_y2))
                            positive_anno_file.flush()
                            face_resize.save(os.path.join(positive_dir, "{}.jpg".format(positive_count)))
                            positive_count += 1
                        elif 0.45 > iou > 0.3:
                            part_anno_file.write("part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, 2, offset_x1, offset_y1, offset_x2, offset_y2))
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_dir, "{}.jpg".format(part_count)))
                            part_count += 1
                        elif iou < 0.2:
                            negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_dir, "{}.jpg".format(negative_count)))
                            negative_count += 1

                        count = positive_count + part_count + negative_count
                    if count > stopvalue:
                        break
            except:
                traceback.print_exc()
    except:
        traceback.print_exc()


if __name__ == '__main__':
    # generate_sample(12, 50000)
    # generate_sample(24, 50000)
    generate_sample(48, 50000)




