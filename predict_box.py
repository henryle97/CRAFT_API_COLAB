import math
import os
import time

import cv2
import torch
from torch.autograd import Variable
from torch.backends import cudnn

from model import craft_utils
from model.craft_utils import copyStateDict
from model.craft import CRAFT
import params as pr
from utils import imgproc, file_utils
from model.refinenet import RefineNet
import numpy as np
from utils.file_utils import displayResult

class CraftDetection:
    def __init__(self):
        self.model = CRAFT()
        if pr.cuda:
            self.model.load_state_dict(copyStateDict(torch.load(pr.trained_model)))
            self.model.cuda()
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = False
        else:
            self.model.load_state_dict(copyStateDict(torch.load(pr.trained_model, map_location='cpu')))
        self.model.eval()

        self.refine_model = None
        if pr.refine:
            self.refine_model = RefineNet()
            if pr.cuda:
                self.refine_model.load_state_dict(copyStateDict(torch.load(pr.refiner_model)))
                self.refine_model = self.refine_net.cuda()
                self.refine_model = torch.nn.DataParallel(self.refine_model)
            else:
                self.refine_model.load_state_dict(copyStateDict(torch.load(pr.refiner_model, map_location='cpu')))

            self.refine_model.eval()
            pr.poly = True

    def text_detect(self, image):
        # if not os.path.exists(image_path):
        #     print("Not exists path")
        #     return []
        # image = imgproc.loadImage(image_path)       # numpy array img (RGB order)
        # image = cv2.imread()

        time0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, pr.canvas_size,
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=pr.mag_ratio)
        print(img_resized.shape)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if pr.cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.model(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if self.refine_model is not None:
            with torch.no_grad():
                y_refiner = self.refine_model(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        time0 = time.time() - time0
        time1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, pr.text_threshold, pr.link_threshold, pr.low_text, pr.poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]


        # expand box: poly  = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)


        if pr.horizontal_mode:
            if self.check_horizontal(polys):
                height, width, channel = image.shape
                new_polys = []
                for box in polys:
                    [[l1, t1], [r1, t2], [r2, b1], [l2, b2]] = box
                    if t1 < t2:
                        l, r, t, b = l2, r1, t1, b1
                    elif t1 > t2:
                        l, r, t, b = l1, r2, t2, b2
                    else:
                        l, r, t, b = l1, r1, t1, b1
                    h_box = abs(b - t)
                    t = max(0, t - h_box * pr.expand_ratio)
                    b = min(b + h_box * pr.expand_ratio, height)
                    x_min, y_min, x_max, y_max = l, t, r, b
                    new_box = [x_min, y_min, x_max, y_max]
                    new_polys.append(new_box)

                polys = np.array(new_polys, dtype=np.float32)

        # for box in polys:

        time1 = time.time() - time1
        total_time = round(time0 + time1, 2)

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if pr.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(time0, time1))
        if pr.folder_test:
            return boxes, polys, ret_score_text

        #if pr.visualize:
            #img_draw = displayResult(img=image[:, :, ::-1], boxes=polys)
            #plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
            #plt.show()

        result_boxes = []
        for box in polys:
            result_boxes.append(box.tolist())
        return result_boxes, total_time


    def check_horizontal(self, boxes):
        total_box = len(boxes)
        num_box_horizontal = 0
        for box in boxes:
            [[l1, t1], [r1, t2], [r2, b1], [l2, b2]] = box
            if t1 == t2:
                num_box_horizontal += 1

        ratio_box_horizontal = num_box_horizontal / float(total_box)
        print("Ratio box horizontal: ", ratio_box_horizontal)
        if ratio_box_horizontal >= pr.ratio_box_horizontal:
            return True
        else:
            return False




if __name__ == "__main__":
    app = CraftDetection()
    # boxes = app.text_detect("test_imgs/qc_6.jpg", visualize=True, horizontal_mode=True)
    # print(boxes)
    # print(boxes)
    # print(score)
    #HST

