import numpy as np


def iou(box, boxes, isMin=False):
    """
    计算iou
    :param box:
    :param boxes:
    :param isMin: 是否是嵌套框
    :return:
    """
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    inter_weight = np.maximum(0, xx2 - xx1)
    inter_height = np.maximum(0, yy2 - yy1)

    inter_area = inter_weight * inter_height
    if isMin:
        resiou = np.divide(inter_area, np.minimum(box_area, boxes_areas))
    else:
        resiou = np.divide(inter_area, (box_area + boxes_areas - inter_area))

    return resiou


def nms(boxes, thresh=0.3, isMin=False):
    """
    非极大值抑制：0个框——>空    1个框——>本身   多个框——>nms去除
    :param boxes: shape(N, 5)
    :param thresh:
    :param isMin:
    :return:
    非极大值抑制
    :param boxes: shape(N,5)   每一行为：（x1, y1, x2, y2, 置信度）
    :param overlap_threshold: 阈值
    :param mode: 'union' or 'min'.
    :return:所选方框的索引组成的列表
    """
    if boxes.shape[0] == 0:
        return np.array([])
    boxes_sort = boxes[(-boxes[:, 4]).argsort()]
    res_boxes = []
    while boxes_sort.shape[0] > 1:
        a_box = boxes_sort[0]
        b_boxes = boxes_sort[1:]
        res_boxes.append(a_box)

        index = np.where(iou(a_box, b_boxes, isMin) < thresh)
        boxes_sort = b_boxes[index]

    if boxes_sort.shape[0] > 0:
        res_boxes.append(boxes_sort[0])
    return np.stack(res_boxes)


def softnms(boxes, thresh_iou=0.3, sigma=0.5, thresh_conf=0.01, isMin=False):
    """
    py_cpu_softnms
    :param cboxes:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param thresh_iou:     iou 的阈值
    :param sigma:  使用 gaussian 函数的方差
    :param thresh_keep: 最后是否保留框的阈值
    :return:       留下的 boxes 的 index
    """
    if boxes.shape[0] == 0:
        return np.array([])
    res_boxes = []
    while boxes.shape[0] > 1:
        boxes_sort = boxes[(-boxes[:, 4]).argsort()]
        a_box = boxes_sort[0]
        b_boxes = boxes_sort[1:]
        res_boxes.append(a_box)

        over = iou(a_box, b_boxes, isMin)
        # iou满足条件的box索引
        index = np.where(over < thresh_iou)
        # iou不满足条件的box索引
        index_keep = np.where(over >= thresh_iou)
        boxes_y = b_boxes[index]
        boxes_no = b_boxes[index_keep]
        boxes_no[:, 4] = boxes_no[:, 4] * np.exp(-np.square(over[index_keep]) / sigma)

        boxes = np.vstack((boxes_y, boxes_no))

    if boxes.shape[0] > 0:
        res_boxes.append(boxes[0])
    _boxes = np.stack(res_boxes)
    res_boxes = _boxes[_boxes[:, 4] > thresh_conf]
    return np.stack(res_boxes)


def convert_to_square(box):
    """
    坐标框转为正方形
    :param box:
    :return:
    """
    square_box = box.copy()
    if box.shape[0] == 0:
        return np.array([])
    weight = box[:, 2] - box[:, 0]
    height = box[:, 3] - box[:, 1]
    max_side = np.maximum(weight, height)

    square_box[:, 0] = box[:, 0] + weight * 0.5 - max_side * 0.5
    square_box[:, 1] = box[:, 1] + height * 0.5 - max_side * 0.5
    square_box[:, 2] = square_box[:, 0] + max_side
    square_box[:, 3] = square_box[:, 1] + max_side
    return square_box


def calibrate_box(bboxes, offsets):
    """
    将边界框转为更像真实的边界框，对边界框进行校正
    :param bboxes: shape(N, 5)
    :param offsets: 网络输出的偏移量 shape(N, 4)
    :return: 修正后的框shape(N, 5)
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1, y2 - y1
    # 在指定的维度上扩维度w(N,1)——>w(N, 1, 1)
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)
    # 在水平方向上扩展, shape(N,4,1)
    translation = np.hstack([w, h, w, h])*offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


if __name__ == '__main__':
    # a = np.array([1, 1, 5, 5, 0.5])
    # b = np.array([[2, 2, 6, 6], [6, 6, 9, 9]])
    # res = iou(a, b)
    # print(res)
    b = np.array([[3, 3, 6,  6, 0.7], [2, 2, 7, 7, 0.8], [1, 1, 4, 4, 0.4]])
    print(b.shape)
    res = nms(b)
    print(res)
    c = np.array([[3, 2, 6, 7, 0.7], [2, 2, 4, 8, 0.8], [1, 4, 8, 9, 0.4]])

    res = convert_to_square(c)
    print(res)
