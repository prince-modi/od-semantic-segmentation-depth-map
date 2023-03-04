from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import numpy as np
import cv2


class PanopticSeg:
    def __init__(self) -> None:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg

    def panoptic_pred_2(self, frame, id1=44, id2=21,  save=False, show=False):
        self.frame = frame
        self.outputs, self.segments_info = self.predictor(frame)["panoptic_seg"]
        # to filter out only the pavement category from the results
        flt1 = [i['id'] for i in self.segments_info if 'category_id' in i and i['category_id']==id1]
        flt2 = [i['id'] for i in self.segments_info if 'category_id' in i and i['category_id']==id2]
        self.outputs[self.outputs!=flt1[0] or self.outputs!=flt2[0]]=0

        if save or show:
            return self.outputs, PanopticSeg.draw_panoptic(self)
        else:
            return self.outputs, None
        
    def panoptic_pred(self, frame, id=0,  save=False, show=False):
        self.frame = frame
        self.outputs, self.segments_info = self.predictor(frame)["panoptic_seg"]
        # to filter out only the pavement category from the results
        flt = [i['id'] for i in self.segments_info if 'category_id' in i and i['category_id']==id]
        self.outputs[self.outputs!=flt[0]]=0

        if save or show:
            return self.outputs, PanopticSeg.draw_panoptic(self)
        else:
            return self.outputs, None

    def draw_panoptic(self) -> np.ndarray:
        self.v = Visualizer(self.frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        self.output = self.v.draw_panoptic_seg_predictions(self.outputs.to("cpu"),self.segments_info)
        frame = self.output.get_image()[:, :, ::-1]
        return frame

if __name__=="__main__":
    obj = PanopticSeg()
    i=0
    while True:
        frame = cv2.imread(f"finale/{i}.jpg")
        output=obj.panoptic_pred(frame)
        cv2.imshow("output",obj.draw_panoptic())
        i += 1
