import cv2
import numpy as np
from PIL import Image


def pil2cv(pilimg):
    pass


def cv2pil(cvimg):
    return Image.fromarray(cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB))


def imread(path, tw, th, mode="PIL"):
    img = cv2.imread(path)
    width, height = img.shape[1], img.shape[0]
    new_w, new_h = int(width*min(tw/width, th/height)), int(height*min(tw/width, th/height))
    canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
    canvas[(th-new_h)//2:(th-new_h)//2+new_h, (tw-new_w)//2:(tw-new_w)//2+new_w, :] = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    if mode == "PIL":
        canvas = cv2pil(canvas)
    elif mode == "CV":
        pass
    else:
        print("mode wrong")
    return canvas, new_w, new_h


if __name__ == '__main__':
    path = "C:/Users/XR/Desktop/traffic/04000~04999/04000.png"
    img, w, h = imread(path, 704, 256, mode="CV")
    cv2.imshow("img", img)
    cv2.waitKey(0)