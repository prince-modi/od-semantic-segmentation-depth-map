from track_git import Detect_track
from depth import Depth
from panoptic import PanopticSeg
import iros_utils as iu
from ultralytics import YOLO

import cv2
import time
import shutil
import os
import glob
from pickle import load, dump

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
yolo_classes_id_dict = yolo.model.names
vest_class_id_dict = vest.model.names
dpt = Depth() #depth model
# pos = PanopticSeg()
dist = load(open('median_dpt.pkl', 'rb'))
print('loaded all models')

# input = "/home/sumanraj/Pictures/ABS_Calc/test/"
# input = "/home/sumanraj/IROS_official/ABS_test"
# input = "/home/sumanraj/IROS_official/ABS_calculation_dataset"
input = "/home/sumanraj/IROS_official/finale/dummy_frames/1"
if input is not None:
    image_names = sorted(glob.glob(os.path.join(input, "*")))
    num_images = len(image_names)
    # print(image_names)

#INFERENCING LOOP
model = YOLO("yolov8x.pt")
with open("REV_results.txt",'w') as f:
    pass 
for i, temp in enumerate(image_names, start=1):
    print(i,temp)
    t1 = time.time()
    frame_path = f"{temp}"
    # frame_path = f"/home/sumanraj/IROS_official/finale/dummy_frames/1/{i+680}.jpg"
    # print(frame_path)
    frame = cv2.imread(frame_path)
    frame = cv2.resize(frame, (960, 720), interpolation = cv2.INTER_AREA)
    
    # try:
    # iu.write_results(frame, frame_path, model,dpt)
    t2=time.time()
    save,show=(False,False)
    vest_results, vest_detections, vest_label = vest.detect_track(frame, i, save, show, send_label=True)
    t3=time.time()
    obs_results, obs_detections, obs_labels  = yolo.detect_track(frame, i, save, show, send_label=True)
    labels = vest_label + obs_labels
    t4=time.time()
    depth = dpt.inference_depth(frame_path, save=False, show=True)
    track_results= iu.calculate_depth(vest_results, obs_results, labels, dist, depth)
    iu.box_annotator(frame, vest_detections, obs_detections, yolo_classes_id_dict, vest_class_id_dict, track_results)
    # print(2)
    # vip_bbox = iu.vip_vest_inter(vest_obj, obs_obj)
    # print(vip_bbox)
    # print(tuple(vip_bbox[0:2]), tuple(vip_bbox[2:4]))
    # frame = cv2.rectangle(frame, tuple(vip_bbox[0:2]), tuple(vip_bbox[2:4]), (255, 0, 0), thickness=3)
    cv2.imshow("frame",frame)
    cv2.waitKey(1000)
    t5=time.time()
    # # segment_obj,img = pos.panoptic_pred(frame, 21, save, show)
    t6=time.time()
    # i += 1
    # except Exception as e:
    #     print(e)
    #     break
    print(f'loop time: {time.time()-t1}, vest: {t3-t2}, yolo:{t4-t3}, depth:{t5-t4}, seg:{t6-t5}\n')
