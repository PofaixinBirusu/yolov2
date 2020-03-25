import torch
from yolov2 import model
from yolov2.detection import YoloDetection
from data import *
import cv2


CAN_USE_GPU = torch.cuda.is_available()
net = model.Darknet19(n_class=3)
if CAN_USE_GPU:
    net = net.cuda()
w, h = 608, 384
param_path = "./param/car.pth"

tensor2img = transforms.ToPILImage()
net.load_state_dict(torch.load(param_path))
colors = [(112, 211, 127), (127, 254, 254), (98, 169, 0)]
class_name = ['car', 'truck', 'person']


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
    for i in range(len(dataset)):
        single_test(dataset.img_list[i].path, thresh=0.3)