#source badri/.venv/bin/activate

#sudo update-alternatives --config python3
#df -h (memory)

#python darknet_images.py --input data/vest.png --weights weights/custom-yolov4-tiny-detector_best.weights --config_file cfg/custom-yolov4-tiny-detector.cfg --data_file cfg/coco.data  

#importing libraries
import cv2
import os, datetime
import time
import shutil
from djitellopy import Tello
import drone_commands as dc
#from drone_commands import 
import darknet
import custom_darknet as detect
import mediapipe as mp
import csv
import numpy as np
import pandas as pd
import pickle
import play_record_video as prv
print("basic libraries loaded")

#mediapipe vars
mp_pose = mp.solutions.pose

def make_output_folder():
	#modifying inference output folder
	out = '/home/dream-nano6/cv_tasks/tracking/darknet-master/bodypose/inference/output'
#	if os.path.exists(out):
#		shutil.rmtree(out)  # delete output folder
	out = os.path.join(out, 'outdoor-testing', datetime.datetime.now().strftime('%H-%M-%S'))
	print(f'created {out}')
	os.makedirs(out)
	print('output folder cleared and created')
	return out

def tello_start(mode):
	tello = Tello()
	if mode == 1 or mode == 2:
		tello.connect()
		print('battery', tello.get_battery())
		tello.streamon()
		frame = tello.get_frame_read().frame
	return tello

def fly_mode_warmup(tello, yolov4_tiny_model, class_names, class_colors, pose_classify_model, pose, mode, i, out):
	if mode == 2:
		# dummy inferencing for warmp
		print('dummy inferencing: gpu-cpu warmup')
		image = tello.get_frame_read().frame
		final_img, bbox_info = detect.main(yolov4_tiny_model, class_names, class_colors, image, out, i)
		image, inf_time, bodypose = dc.pose_detect(image, pose, pose_classify_model)                        
		print('dummy inferencing done!')
		tello.takeoff()
		dc.hover(tello, 130)
#		tello.send_rc_control(0, 0, 30, 0)	
#		time.sleep(3)
#		tello.send_rc_control(0, 0, 0, 0)	
#		time.sleep(1)

def fetch_frame(mode, i, tello):
	if mode == 0:
		image_path = f'/home/dream-nano6/cv_tasks/tracking/yolov4/frames/frame_{i}.png'
		return cv2.imread(image_path) #from system
	else:
		return tello.get_frame_read().frame

def load_yolo():
	#load vest tracking yolov4-tiny model
	cfg = 'cfg/custom-yolov4-tiny-detector.cfg'
	data = 'cfg/coco.data'
	weights = 'weights/custom-yolov4-tiny-detector_best.weights'
	yolov4_tiny_model, class_names, class_colors = darknet.load_network(cfg, data, weights, batch_size=1)
	return yolov4_tiny_model, class_names, class_colors

def load_pose_classify():
	#load pose classification model
	with open('body_language.pkl', 'rb') as f:
		pose_classify_model = pickle.load(f)
	return pose_classify_model 

def end_loop(out, tello, mode):
	cv2.destroyAllWindows()
	ctrl = int(input('Keyboard control? (0:no, 1:yes, 2:ssh): '))
	if mode != 0:
		if ctrl == 1:
			dc.keyboard_control(tello)
		else:
			tello.streamoff()
			tello.land()
	vid_path = '/home/dream-nano6/cv_tasks/tracking/darknet-master'
	prv.video(out, vid_path, 'vest_pose')	
	print("done executing")

def main():
	#setup 
	out = make_output_folder()
	yolov4_tiny_model, class_names, class_colors = load_yolo()
	pose_classify_model = load_pose_classify()
	mode = 2  #0 = system images; 1 = tello image but not fly; 2 = tello Image+fly
	tello = tello_start(mode) #input change to 1 if tello operation		
	
	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
		flag = 0
		try: 
			i=0 #iterator for frames
			fly_mode_warmup(tello, yolov4_tiny_model, class_names, class_colors, pose_classify_model, pose, mode, i, out)
			fb_error, yaw_error, ud_error = 0, 0, 0
			flag, high_h_ctr, low_h_ctr, pose_detect_ctr, go_back_status, bodypose_array = 0, 0, 0, 0, False, []
			while True:
				start = time.time()
				try:
					image = fetch_frame(mode, i, tello)
					i += 1
					pose_detect_ctr += 1
				except Exception as e:
					print('frame not captured', e)
					break
				
				# pose detection and classify
#				image, inf_time, bodypose = dc.pose_detect(image, pose, pose_classify_model)
				if pose_detect_ctr == 3:
					image, inf_time, bodypose = dc.pose_detect(image, pose, pose_classify_model)
					pose_detect_ctr = 0
					bodypose_array, i = dc.pose_based_control(tello, pose, pose_classify_model, bodypose, bodypose_array, i, image, out)
								
				#Hazard vest detection & bbox                          
				final_img, bbox_info = detect.main(yolov4_tiny_model, class_names, class_colors, image, out, i)
				
				resized_img = cv2.resize(final_img, (960, 720), interpolation=cv2.INTER_AREA)
				cv2.imshow('frame', resized_img)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				
				if bbox_info == None:
					print(f'flag: {flag}, high_h_ctr: {high_h_ctr}, low_h_ctr: {low_h_ctr}, go_back_status: {go_back_status}, pose_detect_ctr: {pose_detect_ctr}')
					flag += 1
					if flag >= 15:
						flag -= 2
						if high_h_ctr <= 9:
							dc.rotate_ccw(tello, i, 45, out)
							high_h_ctr += 1
						if high_h_ctr > 9:
							if low_h_ctr == 0:
								dc.lower_height(tello)
								low_h_ctr += 1
								continue
							dc.rotate_ccw(tello, i, 45, out)
							low_h_ctr += 1
							if low_h_ctr >= 9:
								if go_back_status == False:
									dc.go_back(tello)
									go_back_status = True
									continue
								dc.rotate_ccw(tello, i, 45, out)
								low_h_ctr += 1
								if low_h_ctr >= 18:
									dc.call_mavic(tello)
									dc.land(tello)
									break
					print("bbox not detected, better try next time!")
					print(f'total loop time: {time.time()-start}s\n')
					continue 
				if flag >= 5:
					tello.send_rc_control(0, 0, 0, 0)
					time.sleep(1)
				flag, high_h_ctr, low_h_ctr, go_back_status = 0, 0, 0, False				
				
					
				#drone_controls
				fb_vel, yaw_vel, ud_vel, fb_error, yaw_error, ud_error = dc.vestTracking(bbox_info, final_img, fb_error, yaw_error, ud_error, tello) 
				
				if mode == 0:
					print(f'total loop time: {time.time()-start}s\n')
				else:
					print(f'total loop time: {time.time()-start}s, battery: {tello.get_battery()}\n')
			end_loop(out, tello, mode)
		
		except KeyboardInterrupt:
			end_loop(out, tello, mode)
		
if __name__ == '__main__':
	print('executing vest_pose.py')
	main()
	print('program completed successfully')
	
