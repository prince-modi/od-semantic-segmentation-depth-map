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

    ## puts mask on roads and pavements, change both ids to same value if only one of the mask is required
    def panoptic_pred_2(self, frame, id1=44, id2=21,  save=False, show=False):
        self.frame = frame
        self.outputs, self.segments_info = self.predictor(frame)["panoptic_seg"]
        # to filter out only the pavement category from the results
        try:
            flt1 = [i['id'] for i in self.segments_info if 'category_id' in i and i['category_id']==id1]# or i['category_id']==44)]
            flt2 = [i['id'] for i in self.segments_info if 'category_id' in i and i['category_id']==id2]# road or pavement
        except Exception as ee:
            print(ee)
        outputs1=self.outputs.clone().detach()
        outputs2=self.outputs.clone().detach()
        if len(flt1):
            outputs1[outputs1!=flt1[0]]=0
        if len(flt2):
            outputs2[outputs2!=flt2[0]]=0
        if len(flt1) and not len(flt2):
            self.outputs=outputs1
        if len(flt2) and not len(flt2):
            self.outputs=outputs2
        self.outputs=outputs1+outputs2
        if save or show:
            return self.outputs, PanopticSeg.draw_panoptic(self)
        else:
            return self.outputs.to("cpu").numpy(), None
        
    ## base function
    def panoptic_pred(self, frame, save=False, show=False):
        self.frame = frame
        self.outputs, self.segments_info = self.predictor(frame)["panoptic_seg"]

        if save or show:
            return self.outputs, PanopticSeg.draw_panoptic(self)
        else:
            return self.outputs.to("cpu").numpy(), None

    ## puts mask on all the "objects" including the vip
    def panoptic_pred_3(self, frame, save=False, show=False):
        self.frame = frame
        self.outputs, self.segments_info = self.predictor(frame)["panoptic_seg"]
        outputs1=self.outputs.clone().detach()
        try:
            mask = [i['id'] for i in self.segments_info if i['isthing']==True]
            temp = np.isin(outputs1.to("cpu"), mask)
            self.outputs[temp==0]=0
        except Exception as ee:
            print("exception",ee)

        if save or show:
            return self.outputs, PanopticSeg.draw_panoptic(self)
        else:
            return self.outputs.to("cpu").numpy(), None

    def draw_panoptic(self) -> np.ndarray:
        self.v = Visualizer(self.frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        self.output = self.v.draw_panoptic_seg_predictions(self.outputs.to("cpu"),self.segments_info)
        frame = self.output.get_image()[:, :, ::-1]
        return frame

if __name__=="__main__":
    obj = PanopticSeg()
    i=0
    while True:
        frame = cv2.imread(f"/home/sumanraj/IROS_official/finale2/extracted/free_road/{i}.jpg")
        # frame = frame[...,::-1]
        output=obj.panoptic_pred(frame)
        cv2.imshow("output",obj.draw_panoptic())
        cv2.waitKey(1)
        i += 1
