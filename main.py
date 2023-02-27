from track_pm import DetectTrack
from depth import Depth
from panoptic import PanopticSeg
from utlity import save_show

import cv2
import time
import shutil
import os

#CREATING OUTPUT FOLDER
out = "output"
if out is not None:
    shutil.rmtree(out)
    os.makedirs(out + "/yolo", exist_ok=True)
    os.makedirs(out + "/vest", exist_ok=True)
    os.makedirs(out + "/depth", exist_ok=True)
    os.makedirs(out + "/seg", exist_ok=True)


#LOADING MODELS
yolo = DetectTrack("yolo") #obstacle detection & tracking model
vest = DetectTrack("vest") #vest detection & tracking model
dpt = Depth() #depth model
pos = PanopticSeg()
print('loaded all models')

#INFERENCING LOOP
i = 0
while True:
    t1 = time.time()
    input = "demo"
    frame_path = f"{input}/{i}.jpg"
    frame = cv2.imread(frame_path)
    try:
        t2=time.time()
        save,show=(True,False)
        vest_obj,frame = vest.detect_track(frame, save, show)
        save_show(frame, i, save, show)
        t3=time.time()
        save,show=(True,False)
        obs_obj,frame = yolo.detect_track(frame, save, show)
        save_show(frame, i, save, show)
        t4=time.time()
        depth = dpt.inference_depth(frame_path, save=True, show=False)
        t5=time.time()
        save,show=(False,False)
        # if i%2==0:
        #     continue
        segment_obj,frame = pos.panoptic_pred(frame, 21, save, show)
        save_show(frame, i, save, show)
        t6=time.time()
        i += 1
    except Exception as e:
        print(e)
        break
    print(f'loop time: {time.time()-t1}, vest: {t3-t2}, yolo:{t4-t3}, depth:{t5-t4}, seg:{t6-t5}\n')