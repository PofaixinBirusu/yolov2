import torch
from yolov2.box import build_yolo_anchor
from yolov2.box import batch_box_gt_iou
from yolov2.box import yolo2box
from yolov2.box import gt_mask_from_gt
from yolov2.box import range_mask_from_gt
from yolov2.box import shift_gts_to_center
from yolov2.box import box2yolo
from yolov2.box import adjust_anchor
from yolov2.box import nms
from torch import nn

CAN_USE_GPU = torch.cuda.is_available()


class YoloDetection:
    def __init__(self, w=416, h=416, cell_w=32):
        self.w_n, self.h_n, self.cell_w = w//cell_w, h//cell_w, cell_w
        anchors = [
            [4.491486710963459, 3.882007890365448],
            [0.4135800547808781, 0.3978305527888442],
            [0.7791677344951261, 0.5442241158380313],
            [1.1481197546804411, 0.9305196901226608],
            [2.137198240589198, 1.5242430441898531]
        ]
        self.yolo_anchors = build_yolo_anchor(anchors, self.w_n*self.h_n)
        self.cell_anchor_num = len(anchors)
        self.anchor_num = self.yolo_anchors.shape[0]

    def loss(self, offset_pred, confidence_pred, cls_pred, gts, cls):
        # 先计算anchors与中心gt的iou
        batch_size, gt_num = gts.shape[0], gts.shape[1]
        iou = batch_box_gt_iou(
            yolo2box(self.yolo_anchors, w_n=self.w_n)
                .contiguous().view(1, self.anchor_num, 4)
                .expand_as(torch.empty(batch_size, self.anchor_num, 4)),
            shift_gts_to_center(gts)
        )
        # 负样本标签
        label = torch.zeros(batch_size, self.anchor_num)
        if CAN_USE_GPU:
            label = label.cuda()
        label[iou.max(2)[0] < 0.6] = 1
        gt_mask = gt_mask_from_gt(gts)
        _, pos_anchor_indexes = (iou.permute([0, 2, 1])*range_mask_from_gt(gts, self.anchor_num, self.w_n)).max(2)
        # 把与gt有最大iou的anchor改成正的标签
        for i in range(batch_size):
            label[i, pos_anchor_indexes[i][gt_mask[i] == 1]] = 0
        # 把pos_indexes对应的预测值拿出来
        pos_confidence_pred = torch.cat([confidence_pred[i, pos_anchor_indexes[i]] for i in range(batch_size)], dim=0)
        pos_cls_pred = torch.cat([cls_pred[i, pos_anchor_indexes[i]] for i in range(batch_size)], dim=0)
        pos_draw = torch.cat([yolo2box(self.yolo_anchors, w_n=self.w_n)[pos_anchor_indexes[i]] for i in range(batch_size)], dim=0)
        # 把offset的真值构造出来，正框的真值是对应的gt，其他是对应的anchor
        offset_target = torch.zeros(batch_size, self.anchor_num, 4)
        offset_weight = torch.zeros(batch_size, self.anchor_num, 1)
        if CAN_USE_GPU:
            offset_target, offset_weight = offset_target.cuda(), offset_weight.cuda()
        offset_target[:, :, 0:2] = 0.5
        offset_weight[:, :, 0:1] = 0.01
        for i in range(batch_size):
            valid_gt = box2yolo(gts[i][gt_mask[i] == 1])
            gt_w, gt_h = valid_gt[:, 2:3]/self.w_n, valid_gt[:, 3:4]/self.h_n
            # 对应的anchor
            target_anchor = self.yolo_anchors[pos_anchor_indexes[i][gt_mask[i] == 1], :]
            valid_gt[:, 2:4] = torch.log(valid_gt[:, 2:4]/target_anchor[:, 2:4])
            # 正样本的权重是2-gt_w*gt_h 越小的框惩罚越大
            offset_target[i, pos_anchor_indexes[i][gt_mask[i] == 1], :] = valid_gt
            offset_weight[i, pos_anchor_indexes[i][gt_mask[i] == 1], :] = 2-gt_w*gt_h
        # confidence损失 负样本跟0做，正样本跟1做
        confidence_loss = (((0-confidence_pred.view(-1))**2)*label.view(-1)).sum()+\
                          (((1-pos_confidence_pred)**2).view(-1)*gt_mask.view(-1)).sum()
        # offset损失
        offset_loss = (((offset_pred-offset_target)**2)*offset_weight).sum()
        # 分类损失 用BCELoss
        cls = cls.view(-1, cls.shape[2])
        cls_loss = ((-(cls*torch.log(pos_cls_pred))-((1-cls)*torch.log(1-pos_cls_pred))).sum(1)*gt_mask.view(-1)).sum()
        # print("confidence_loss: %.3f  offset_loss: %.3f  cls_loss: %.3f" %
        #       (confidence_loss.item(), offset_loss.item(), cls_loss.item()))
        return confidence_loss+offset_loss+cls_loss, pos_draw, pos_cls_pred, pos_confidence_pred

    def select_boxes(self, offset_pred, confidence_pred, cls_pred, confidence_thresh=0.7):
        box_pred = yolo2box(adjust_anchor(self.yolo_anchors, offset_pred[0]), w_n=self.w_n)
        confidence_pred, cls_pred = confidence_pred.view(-1), cls_pred[0].argmax(1)
        boxes, confidence, cls = box_pred[confidence_pred > confidence_thresh], confidence_pred[confidence_pred > confidence_thresh], cls_pred[confidence_pred > confidence_thresh]
        boxes, confidence, cls = boxes.cpu(), confidence.cpu(), cls.cpu()
        label = nms(boxes, confidence, thresh=0.4)
        return boxes[label == 0], confidence[label == 0], cls[label == 0]


if __name__ == '__main__':
    box_target = torch.zeros(3, 5, 4)
    box_target[:, :, 0:2] = 0.5
    box_weight = torch.ones(3, 5, 1)
    gt = [
        [[1, 2, 3, 4],
         [3, 4, 5, 6]],
        [[6, 6, 6, 6],
         [8, 8, 8, 8]],
        [[1, 1, 4, 4],
         [2, 2, 3, 3]]
    ]
    pos_indexes = [
        [1, 4],
        [2, 3],
        [0, 2]
    ]
    gt_mask = [
        [1, 1],
        [1, 0],
        [0, 1]
    ]
    gt, pos_indexes, gt_mask = torch.Tensor(gt), torch.Tensor(pos_indexes).long(), torch.Tensor(gt_mask)

    for i in range(3):
        valid_gt = gt[i][gt_mask[i] == 1]
        box_target[i, pos_indexes[i][gt_mask[i] == 1], :] = valid_gt
        box_weight[i, pos_indexes[i][gt_mask[i] == 1], :] = valid_gt[:, 2:3]*valid_gt[:, 3:4]
    print(box_target)
    print(box_weight)
    box = [
        [2, 2],
        [3, 4]
    ]
    weight = [[1], [2]]
    box, weight = torch.Tensor(box), torch.Tensor(weight)
    print(box*weight)
    pred = [
        [0.6, 0.4, 0.2, 0.3],
        [0.3, 0.2, 0.6, 0.7]
    ]
    real = [
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]
    pred, real = torch.Tensor(pred), torch.Tensor(real)
    loss_fn = nn.BCELoss(reduction="sum")
    import math
    def ln(x):
        return math.log(x, math.e)
    print(ln(0.4)+ln(0.4)+ln(0.8)+ln(0.7))
    print(ln(0.7)+ln(0.8)+ln(0.4)+ln(0.7))
    print((-(real*torch.log(pred))-((1-real)*torch.log(1-pred))).sum(1))
    print(loss_fn(pred, real))
