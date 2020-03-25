import torch
from yolov2 import model
from yolov2.detection import YoloDetection
from data import *
import cv2
import random
from yolov2.yolo_util import imread
import numpy as np


CAN_USE_GPU = torch.cuda.is_available()
net = model.Darknet19(n_class=3)
if CAN_USE_GPU:
    net = net.cuda()

w, h = 608, 384
batch_size = 1
epoch, learning_rate = 100, 0.00001
param_path = "./param/car.pth"

yolo = YoloDetection(w=w, h=h)
tensor2img = transforms.ToPILImage()

net.load_state_dict(torch.load(param_path))
colors = [(112, 211, 127), (127, 254, 254), (98, 169, 0)]
class_name = ['car', 'truck', 'person']


def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


def draw_rec(img, box, color=(0, 255, 0)):
    xmin, ymin, xmax, ymax = box
    return cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)


def single_test(test_path, thresh=0.8):
    net.eval()
    yolo = YoloDetection(w=w, h=h)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    with torch.no_grad():
        cv_img0 = imread(test_path, tw=w, th=h, mode="CV")[0]
        x = transform(imread(test_path, tw=w, th=h, mode="PIL")[0])
        if CAN_USE_GPU:
            x = x.cuda()
        box_pred, confidence, class_pred = net(x.unsqueeze(0))
        selected_anchor, selected_confidence, cls = yolo.select_boxes(box_pred, confidence, class_pred, confidence_thresh=thresh)
        print(selected_confidence)

        for i, box in enumerate(selected_anchor):
            class_index = cls[i].item()
            con_val = selected_confidence[i].item()
            color = colors[class_index]
            xmin, ymin = box.detach().numpy()[0], box.detach().numpy()[1]
            cv_img0 = draw_rec(cv_img0, list(box.detach().numpy()), color)
            cv_img0 = cv2.putText(cv_img0, "%s %.1f" % (class_name[class_index], con_val), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("test", cv_img0)
        cv2.waitKey(0)


if __name__ == '__main__':
    root_path = "C:/Users/XR/Desktop/object-detection-crowdai/"
    label_path = "labels.csv"
    dataset = AutoDriveDataset(root_path, label_path, tw=w, th=h)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    for epoch_count in range(epoch):
        # net.train()
        # loss_val, process = 0, 0
        # for x, gts, cls in dataloader:
        #     torch.cuda.empty_cache()
        #     x, gts, cls = x.cuda(), gts.cuda(), cls.cuda()
        #     box_pred, confidence_pred, class_pred = net(x)
        #     loss, pos_box, cls_p, conf = yolo.loss(box_pred, confidence_pred, class_pred, gts, cls)
        #     torch.cuda.empty_cache()
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     loss_val += loss.item()
        #
        #     process += x.shape[0]
        #     print("\r进度：%s  本批loss:%.5f" % (processbar(process, len(dataset)), loss.item()), end="")

            # 看下框画对了不
            # test_img_num = 0
            # img = cv2.cvtColor(np.asarray(tensor2img(x[test_img_num].cpu())), cv2.COLOR_RGB2BGR)
            # for i in range(len(gts[test_img_num])):
            #     if (gts[test_img_num][i].eq(torch.Tensor([0, 0, 0, 0]).cuda())).sum() == 4:
            #         break
            #     box = [gts[test_img_num][i][0].item(), gts[test_img_num][i][1].item(),
            #            gts[test_img_num][i][2].item(), gts[test_img_num][i][3].item()]
            #     cls_real = cls[test_img_num][i].argmax(0).item()
            #     img = cv2.putText(img, class_name[cls_real], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[cls_real], 2)
            #     img = draw_rec(img, box=box, color=colors[cls_real])
            #     print(pos_box)
            #     box = [pos_box[test_img_num*dataset.max_gt_num+i, 0].item(), pos_box[test_img_num*dataset.max_gt_num+i, 1].item(),
            #            pos_box[test_img_num*dataset.max_gt_num+i, 2].item(), pos_box[test_img_num*dataset.max_gt_num+i, 3].item()]
            #     img = draw_rec(img, box=box, color=(0, 0, 255))
            #     cls_pred = cls_p[test_img_num * dataset.max_gt_num + i].argmax(0).item()
            #     con_val = conf[test_img_num*dataset.max_gt_num+i].item()
            #     img = cv2.putText(img, "%s %.1f" % (class_name[cls_pred], con_val), (int(box[0]), int(box[1])),
            #                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #                       (0, 0, 255), 2)
            # cv2.imshow("pos", img)
            # cv2.waitKey(0)

        # print("\nepoch:%d  loss:%.3f" % (epoch_count+1, loss_val))
        if (epoch_count+1) % 1 == 0:
            test_path = random.choice(dataset.img_list).path
            print(test_path)
            # torch.save(net.state_dict(), param_path)
            single_test(test_path, thresh=0.3)