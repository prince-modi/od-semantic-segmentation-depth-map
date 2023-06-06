from track_git import Detect_track
from depth import Depth
from panoptic import PanopticSeg
from utlity import save_show
import iros_utils as iu
from ultralytics import YOLO

import cv2
import time
import shutil
import os
import glob

#CREATING OUTPUT FOLDER
out = "output"
if out is not None:
    shutil.rmtree(out)
    os.makedirs(out + "/yolo", exist_ok=True)
    os.makedirs(out + "/vest", exist_ok=True)
    os.makedirs(out + "/depth", exist_ok=True)
    os.makedirs(out + "/seg", exist_ok=True)


#LOADING MODELS
yolo = Detect_track("yolo") #obstacle detection & tracking model
vest = Detect_track("vest") #vest detection & tracking model
dpt = Depth() #depth model
# pos = PanopticSeg()
print('loaded all models')

# input = "/home/sumanraj/Pictures/ABS_Calc/test/"
input = "/home/sumanraj/IROS_official/ABS_test"
if input is not None:
    image_names = sorted(glob.glob(os.path.join(input, "*")))
    num_images = len(image_names)
    print(image_names)

#INFERENCING LOOP
model = YOLO("yolov8x.pt")
with open("REV_results.txt",'w') as f:
    pass 
for i, temp in enumerate(image_names, start=1):
    print(i,temp)
    t1 = time.time()
    frame_path = f"{temp}"
    frame = cv2.imread(frame_path)
    frame = cv2.resize(frame, (960, 720), interpolation = cv2.INTER_AREA)
    # cv2.imshow("frame",frame)
    # cv2.waitKey(1)
    # continue
    
    try:
        # iu.write_results(frame, frame_path, model,dpt)
        t2=time.time()
        save,show=(False,False)
        vest_obj = vest.detect_track(frame, i, save, show)
        t3=time.time()
        obs_obj  = yolo.detect_track(frame, i, save, show)
        t4=time.time()
        depth = dpt.inference_depth(frame_path, save=False, show=True)
        obs_dist = iu.calculate_depth(vest_obj, obs_obj, depth)
        t5=time.time()
        # # segment_obj,img = pos.panoptic_pred(frame, 21, save, show)
        t6=time.time()
        ##vip 
        # i += 1
    except Exception as e:
        print(e)
        break
    print(f'loop time: {time.time()-t1}, vest: {t3-t2}, yolo:{t4-t3}, depth:{t5-t4}, seg:{t6-t5}\n')
