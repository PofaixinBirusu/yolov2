import torch
import numpy as np

CAN_USE_GPU = torch.cuda.is_available()


def box2yolo(boxes, cell_w=32):
    xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x, y, w, h = (xmin+xmax+1)/(2*cell_w), (ymin+ymax+1)/(2*cell_w), (xmax-xmin+1)/cell_w, (ymax-ymin+1)/cell_w
    return torch.stack([x-x.int(), y-y.int(), w, h], dim=0).t()


def yolo2box(boxes, w_n=13, cell_w=32, cell_box_num=5):
    box_num = boxes.shape[0]
    box_index = torch.arange(0, box_num)//cell_box_num
    shift_x, shift_y = box_index % w_n, box_index // w_n
    if CAN_USE_GPU:
        shift_x, shift_y = shift_x.cuda(), shift_y.cuda()
    x, y, w, h = boxes[:, 0]*cell_w+shift_x*cell_w, boxes[:, 1]*cell_w+shift_y*cell_w, boxes[:, 2]*cell_w, boxes[:, 3]*cell_w
    return torch.stack([x-w/2, y-h/2, x+w/2, y+h/2]).t()


def batch_iou(boxes1, boxes2):
    xmin1, ymin1, xmax1, ymax1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    xmin2, ymin2, xmax2, ymax2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    lx, ly = torch.stack([xmin1, xmin2], dim=0).max(0)[0], torch.stack([ymin1, ymin2], dim=0).max(0)[0]
    rx, ry = torch.stack([xmax1, xmax2], dim=0).min(0)[0], torch.stack([ymax1, ymax2], dim=0).min(0)[0]
    box_mask = (lx < rx) & (ly < ry)
    jiao_ji = (rx-lx+1)*(ry-ly+1)
    bing_ji = (xmax1-xmin1+1)*(ymax1-ymin1+1)+(xmax2-xmin2+1)*(ymax2-ymin2+1)-jiao_ji
    return (jiao_ji/bing_ji)*box_mask.int()


def batch_box_gt_iou(boxes, gts):
    batch_size, box_num, gt_num = boxes.shape[0], boxes.shape[1], gts.shape[1]
    return batch_iou(
        boxes.contiguous().view(batch_size, box_num, 1, 4)
            .expand_as(torch.empty(batch_size, box_num, gt_num, 4)).contiguous().view(-1, 4),
        gts.contiguous().view(batch_size, 1, gt_num, 4)
            .expand_as(torch.empty(batch_size, box_num, gt_num, 4)).contiguous().view(-1, 4)
    ).view(batch_size, box_num, gt_num)


def range_mask_from_gt(gts, anchor_num, w_n, cell_w=32, cell_anchor_num=5):
    batch_size, gt_num = gts.shape[0], gts.shape[1]
    gt_stk = gts.view(-1, 4)
    shify_x, shift_y = ((gt_stk[:, 0]+gt_stk[:, 2]+1)/(2*cell_w)).int(), ((gt_stk[:, 1]+gt_stk[:, 3]+1)/(2*cell_w)).int()
    start_index = ((shift_y*w_n+shify_x)*cell_anchor_num).cpu()
    range_mask = np.zeros(shape=(gt_stk.shape[0], anchor_num))
    for i in range(gt_stk.shape[0]):
        start = start_index[i].item()
        range_mask[i, start:start+cell_anchor_num] = 1
    range_mask = torch.Tensor(range_mask).contiguous().view(batch_size, gt_num, anchor_num)
    if CAN_USE_GPU:
        range_mask = range_mask.cuda()
    return range_mask


def gt_mask_from_gt(gts):
    batch_size, gt_num = gts.shape[0], gts.shape[1]
    gt_stk = gts.view(-1, 4)
    invalid_gt = torch.zeros(gt_stk.shape[0], 4)
    if CAN_USE_GPU:
        invalid_gt = invalid_gt.cuda()
    return (torch.eq(gt_stk, invalid_gt).sum(1) != 4).float().view(batch_size, gt_num)


def shift_gts_to_center(gts, cell_w=32):
    batch_size, gt_num = gts.shape[0], gts.shape[1]
    gt_stk = gts.view(-1, 4)
    xmin, ymin, xmax, ymax = gt_stk[:, 0], gt_stk[:, 1], gt_stk[:, 2], gt_stk[:, 3]
    x, y = (xmin+xmax+1)/(2*cell_w), (ymin+ymax+1)/(2*cell_w)
    shift_x, shift_y = (0.5-x+x.int())*cell_w, (0.5-y+y.int())*cell_w
    return torch.stack([xmin+shift_x, ymin+shift_y, xmax+shift_x, ymax+shift_y], dim=0).t().view(batch_size, gt_num, 4)


# yolo格式的anchor, [0.5, 0.5, w, h]
def adjust_anchor(anchors, offset_pred):
    return torch.cat([offset_pred[:, 0:2], torch.exp(offset_pred[:, 2:4])*anchors[:, 2:4]], dim=1)


def build_yolo_anchor(anchors, cell_num):
    yolo_anchors = torch.Tensor([[0.5, 0.5, w, h] for w, h in anchors]*cell_num)
    if CAN_USE_GPU:
        yolo_anchors = yolo_anchors.cuda()
    return yolo_anchors


def two_box_iou(box1, box2):
    x1, y1, x2, y2 = box1[0].item(), box1[1].item(), box1[2].item(), box1[3].item()
    x1_, y1_, x2_, y2_ = box2[0].item(), box2[1].item(), box2[2].item(), box2[3].item()
    lx, ly = max(x1, x1_), max(y1, y1_)
    rx, ry = min(x2, x2_), min(y2, y2_)
    if lx < rx and ly < ry:
        jiao_ji = (ry - ly + 1) * (rx - lx + 1)
        bing_ji = (x2 - x1 + 1) * (y2 - y1 + 1) + (x2_ - x1_ + 1) * (y2_ - y1_ + 1) - jiao_ji
        return jiao_ji / bing_ji
    else:
        return 0


def nms(boxes, sorce, thresh=0.5):
    label = torch.ones(size=(boxes.shape[0],))
    iou = [[two_box_iou(box, base) for box in boxes] for base in boxes]
    while (label == 1).sum() > 0:
        v, index = (sorce*label).topk(1, dim=0)
        label[index.item()] = 0
        for i in range(boxes.shape[0]):
            if i != index.item() and iou[index.item()][i] > thresh:
                label[i] = -1
    return label


if __name__ == '__main__':
    box = [
        [0, 0, 31, 31],
        [32, 0, 63, 31]
    ]
    print(box2yolo((torch.Tensor(box))))
    box = [
        [0.5, 0.5, 1, 1]
    ]
    box = torch.Tensor(box*18)
    print(yolo2box(box, 3, cell_box_num=3))
    x = [True, False, True]
    y = [False, False, True]
    x, y, = torch.BoolTensor(x), torch.BoolTensor(y)
    x.contiguous()
    print(x & y)
    box1 = [
        [0, 0, 15, 15],
        [0, 0, 15, 15],
        [0, 0, 15, 15],
        [0, 0, 15, 15],
        [0, 0, 15, 15]
    ]
    box2 = [
        [0, 0, 15, 15],
        [0, 0, 16, 16],
        [1, 1, 16, 16],
        [16, 16, 32, 32],
        [8, 8, 23, 23]
    ]
    box1, box2 = torch.Tensor(box1), torch.Tensor(box2)
    print(batch_iou(box1, box2))
    boxes = [
        [[0, 0, 15, 15],
         [0, 0, 16, 16]],
        [[1, 1, 18, 18],
         [2, 2, 19, 19]]
    ]
    gts = [
        [[0, 0, 16, 16],
         [0, 0, 16, 16],
         [0, 0, 0, 0]],
        [[0, 0, 16, 16],
         [2, 2, 19, 19],
         [2, 2, 38, 38]]
    ]
    boxes, gts = torch.Tensor(boxes), torch.Tensor(gts)
    print(batch_box_gt_iou(boxes, gts))
    gts = [
        [[0, 32, 31, 63],
         [32, 64, 63, 95]],
        [[0, 0, 31, 31],
         [32, 0, 63, 31]]
    ]
    gts = torch.Tensor(gts)
    print(range_mask_from_gt(gts, anchor_num=30, w_n=2, cell_anchor_num=5))

    gts = [
        [[0, 32, 31, 63],
         [37, -5, 68, 26]],
        [[0, 0, 31, 31],
         [0, 0, 0, 0]]
    ]
    gts = torch.Tensor(gts).cuda()
    print(gt_mask_from_gt(gts))

    gts = [
        [[-5, -5, 26, 26],
         [37, -5, 68, 26]],
        [[-5, 37, 26, 68],
         [37, 37, 68, 68]]
    ]
    gts = torch.Tensor(gts)
    print(shift_gts_to_center(gts))

    anchors = [
        [0.5, 0.5, 1, 1],
        [0.5, 0.5, 2, 2],
        [0.5, 0.5, 1.5, 1.5],
        [0.5, 0.5, 3, 4]
    ]
    offset_pred = [
        [0.8, 0.2, 1, 1],
        [0.4, 0.3, 0, 0],
        [0.8, 0.2, 1, 1],
        [0.4, 0.3, 0, 0],
    ]
    print(adjust_anchor(torch.Tensor(anchors), torch.Tensor(offset_pred)))

    x = [
        [1, 2, 3, 4],
        [2, 3, 4, 5]
    ]
    x = torch.Tensor(x)
    print(id(x[0]))
    print(id(x[0, :]))