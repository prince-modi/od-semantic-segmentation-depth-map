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
import numpy as np

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
pos = PanopticSeg()
dist = load(open('median_dpt.pkl', 'rb'))
print('loaded all models')

# input = "/home/sumanraj/Pictures/ABS_Calc/test/"
# input = "/home/sumanraj/IROS_official/ABS_test"
# input = "/home/sumanraj/IROS_official/ABS_calculation_dataset"
# input = "/home/sumanraj/IROS_official/finale/dummy_frames/2"
# input = "/home/sumanraj/IROS_official/finale"
# if input is not None:
    # image_names = sorted(glob.glob(os.path.join(input, "*")))
    # num_images = len(image_names)
    # print(image_names)

#INFERENCING LOOP
model = YOLO("yolov8x.pt")
with open("REV_results.txt",'w') as f:
    pass 
# for i, temp in enumerate(image_names, start=0):
vip_tracker_id = 1000
vip_bbox = [0,0,960,720]
i=10
while True:
    print(i)
    t1 = time.time()
    # frame_path = f"{temp}"
    # frame_path = f"/home/sumanraj/IROS_official/finale/dummy_frames/5/{i}.jpg"
    frame_path =f"/home/sumanraj/IROS_official/finale2/extracted/free_road/{i}.jpg"
    # frame_path = f"/home/sumanraj/IROS_official/demo/{i}.jpg"
    # print(frame_path)
    frame = cv2.imread(frame_path)
    frame = cv2.resize(frame, (960, 720), interpolation = cv2.INTER_AREA)
    
    # try:
    # iu.write_results(frame, frame_path, model,dpt)
    t2=time.time()
    save,show=(False,False)
    vest_results, vest_detections, vest_label = vest.detect_track(frame, i, save, show, send_label=True)
    # vest_obj = vest.detect_track(frame, i, save, show, send_label=False)
    t3=time.time()
    obs_results, obs_detections, obs_labels  = yolo.detect_track(frame, i, save, show, send_label=True)
    # obs_obj = yolo.detect_track(frame, i ,save, show, send_label=False)
    # labels = vest_label + obs_labels
    t4=time.time()
    depth = dpt.inference_depth(frame_path, save=False, show=False)
    track_results= iu.calculate_depth(vest_results, obs_results, vest_label, obs_labels, dist, depth)
    # iu.box_annotator(frame, vest_detections, obs_detections, yolo_classes_id_dict, vest_class_id_dict, track_results)
    vip_bbox, vip_tracker_id = iu.track_vip(obs_detections, track_results, vest_results, obs_results, vip_bbox, vip_tracker_id)
    red_obs_bbox = iu.classify_obs_by_dist(frame, track_results, vip_bbox, threshold=6)
    mask = pos.panoptic_pred_2(frame,id1=21,id2=21)
    iu.vip_close_to_edge(vip_bbox,mask[0], frame)
    frame = cv2.rectangle(frame, tuple(vip_bbox[0:2]), tuple(vip_bbox[2:4]), (255, 0, 0), thickness=6)
    cv2.imshow("frame",frame)
    cv2.imwrite(f"/home/sumanraj/IROS_official/finale2/edge/free_road/main/{i}.jpg",frame)
    cv2.waitKey(1)
    t5=time.time()
    # # segment_obj,img = pos.panoptic_pred(frame, 21, save, show)
    t6=time.time()
    i += 1
    # except Exception as e:
    #     print(e)
    #     break
    print(f'loop time: {time.time()-t1}, vest: {t3-t2}, yolo:{t4-t3}, depth:{t5-t4}, seg:{t6-t5}\n')