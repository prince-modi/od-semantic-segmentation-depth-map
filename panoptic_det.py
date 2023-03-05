import torch, detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger

import numpy as np
import time
import os, json, cv2, random, csv
import shutil

import matplotlib.pyplot as plt
from PIL import Image


cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
predictor = DefaultPredictor(cfg)

if os.path.exists("mask_plots"):
    shutil.rmtree("mask_plots")
os.makedirs("mask_plots")
 
try:
    i = 0
    inf_row = []
    while i<650:
        print(i)
        t0 = time.time()
        frame = cv2.imread(f"demo/{i}.jpg")
        print("read image")
        frame = cv2.resize(frame, (960,720), interpolation=cv2.INTER_AREA)
        t1=time.time()
        outputs, segments_info = predictor(frame)["panoptic_seg"]
        print(1)
        res=time.time()
        # try:
        #     # print("tryyy")
        #     # segs = [i['id'] for i in segments_info if 'category_id' in i and i['category_id']==21]# or i['category_id']==44)]
        #     # print(len(segs))
        #     segs2 = [i['id'] for i in segments_info if 'category_id' in i and i['category_id']==21]# road or pavement
        #     print(len(segs2))
        # except Exception as ee:
        #     continue
        # finally:
        #     i+=1
        # # outputs1=outputs.clone().detach()
        # outputs2=outputs.clone().detach()
        # t2=time.time()
        mask = [i['id'] for i in segments_info if i['isthing']==True]
        # print(mask)
        # outputs[outputs not in mask]=0
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # # if len(segs):
        # #     outputs1[outputs1!=segs[0]]=0
        # if len(segs2):
        #     outputs2[outputs2!=segs2[0]]=0
        # # if len(segs) and not len(segs2):
        # #     finaloutput=outputs1
        # # if len(segs2) and not len(segs):
        # #     finaloutput=outputs2
        # # finaloutput=outputs1+outputs2
        # finaloutput=outputs2

        # with open('output.txt','w') as f:
        #     f.write(str(segments_info))

        out = v.draw_panoptic_seg_predictions(outputs.to("cpu"), segments_info)
        t3 = time.time()
        frame = out.get_image()[:, :, ::-1]
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        cv2.imwrite(f'finale/sem_seg_lane/{i}.jpg', frame) 
        i=i+1
        # print(f'loop: {time.time()-t0}, preprocessing: {t1-t0}, inferencing: {t2-t1}, overhead: {t3-t2}')
        inf_row.append(res-t1)
    with open('panoptic.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SNo", "Time"])
        for i ,row in enumerate(inf_row):
            writer.writerow([i,row])
except Exception as e:
    print(e)

