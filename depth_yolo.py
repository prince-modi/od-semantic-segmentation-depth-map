#pseudo code
import iros_utils as iu
import extract_frames as ef
from track import Detect_track
from depth import Depth
from panoptic import PanopticSeg
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
yolo = Detect_track("yolo") #obstacle detection & tracking model
vest = Detect_track("vest") #vest detection & tracking model
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
        # vest_obj = vest.detect_track(frame, i, save=True, show=True)
        t3=time.time()
        # obs_obj = yolo.detect_track(frame, i, save=True, show=True)
        t4=time.time()
        # depth = dpt.inference_depth(frame_path, save=True, show=False)
        t5=time.time()
        segment = pos.panoptic_pred(frame, i, 44, save=True, show=False)
        print(segment.shape)
        t6=time.time()
        i += 1
    except Exception as e:
        print(e)
        break
    print(f'loop time: {time.time()-t1}, vest: {t3-t2}, yolo:{t4-t3}, depth:{t5-t4}, seg:{t6-t5}\n')










# #segmentation model
# m2f = test.Mask2Former()
# predictor=m2f.load_model()
# i = 0
# while True:
#     t1=time.time()
#     frame = ef.fetch_frame(i)
#     t2=time.time()
#     cv2.imshow("frame", frame)
#     output = predictor(frame)
#     t3=time.time()
    
#     # cv2.imshow("frame", output)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
#     i += 1
#     t4=time.time()
#     print(f't2-t1:{t2-t1}, t3-t2:{t3-t2}. t4-t1:{t4-t1}')



# frame = iu.fetch_frame()
# bboxes = yolo(frame)
# depth = midas(frame)
# rev_arr = iu.relative_dist(bboxes, depth)

#rev_arr contains obj class, id, dist, bbox coord
