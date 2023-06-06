import cv2
import time
import sys
import numpy as np
import pandas as pd
import mediapipe as mp
import play_record_video as prv
from djitellopy import Tello

def tello_start():
	tello = Tello()
	tello.connect()
	print('battery', tello.get_battery())
	tello.streamon()
	image = tello.get_frame_read().frame
	print('tello initiated....')
	return tello

def tello_first_takeoff(tello):
	tello.takeoff()
	tello.send_rc_control(0, 0, 30, 0)	
	time.sleep(3)	

def vestTracking(bbox_info, frame, fb_error, yaw_error, ud_error, tello):
	area = bbox_info[2] #area of bounding box
	x,y = bbox_info[0], bbox_info[1] #coordinates of centre of bounding box
	(h, w) = frame.shape[:2] #W = 960 #H = 720 #Area_frame = 691200
	Initial_height = 50
#	fbRange = [20000, 25000]
	fbRange = [15000, 20000]

	#PID Coefficients	
	yaw_pid = [0.32,0.022,0] #coefficients for yaw velocity
	fb_pid = [0.003, 0.003] #coefficients for front back velocity
	ud_pid = [0.2,0.1] #for up-down velocity

	if area != None:
		#for forward-backward velocity
		if x > 0: 
			if(area>fbRange[0] and area<fbRange[1]):
				err2 = 0
			elif(area < fbRange[0] ):  
				err2 = fbRange[0] - area 
			elif(area > fbRange[1]):
				err2 = fbRange[1] - area
		if x == 0:
			# td = 0
			speed = 0
			fb = 0
			err2 = 0
		fb = fb_pid[0]*err2 + fb_pid[1]*(err2-fb_error)
		fb = int(np.clip(fb, -100, 100))
		fb_error = err2
		
		#for yaw_velocity
		err1 = x - w//2
		speed = yaw_pid[0] * err1 + yaw_pid[1] * (err1-yaw_error)
		speed = int(np.clip(speed, -100, 100))
		yaw_error = err1

		#for up-down velocity
		err3 = -(y - h//2) 
		ud = ud_pid[0]*err3 + ud_pid[1]*(err3-ud_error)
		ud = int(np.clip(ud, -100, 100))
		ud_error = err3

		tello.send_rc_control(0, fb, ud, speed)
		return fb, speed, ud, fb_error, yaw_error, ud_error
	else:
		return 0, 0, 0, fb_error, yaw_error, ud_error

def pose_detect(image, pose, pose_classify_model):
	#mediapipe vars
	mp_drawing = mp.solutions.drawing_utils
	mp_pose = mp.solutions.pose

	#inference
	image.flags.writeable = False
	t1 = time.time()
	results = pose.process(image)
	inf_time = time.time()-t1
	image.flags.writeable = True
	mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1))
                             
    #pose classify
	try:
		# Extract Pose landmarks
		pose_coords = results.pose_landmarks.landmark
		pose_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose_coords]).flatten()) 
		X = pd.DataFrame([pose_row])
		body_language_class = pose_classify_model.predict(X)[0]
		cv2.putText(image, body_language_class, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (234, 221, 202), 3, cv2.LINE_AA) 
	except Exception as e:
		body_language_class = 'None'
		print('Exception', e)
		pass
	return image, inf_time, body_language_class

def rotate_ccw(tello, i=0, angle=0, out = 'inference'):
	print(f'taking left turn {angle} degrees....')
	image = tello.get_frame_read().frame
#	cv2.imshow('image', image)
#	cv2.waitKey(500)
	cv2.imwrite(f'{out}/{i}.png', image)
	tello.rotate_counter_clockwise(angle)
	print('left turn complete!')
					
def lower_height(tello):
	print('lowering height....')
	desired_h = 30
	current_h = tello.get_height()
	prev_err = 0
	ud_pid = [0.2,0.1] #up-down pid constants
	while current_h > 40:
		current_h = tello.get_height()
		err = (desired_h - current_h)*3
		ud = ud_pid[0]*err + ud_pid[1]*(err - prev_err)
		ud = int(np.clip(ud, -100, 100))
		prev_err = err
		tello.send_rc_control(0, 0, ud, 0)
		time.sleep(0.05)
		print(f'current_h: {current_h}')
	print(f'lowered height to {tello.get_height()} cms')
	
def hover(tello, desired_h):
	print('hovering at height 150....')
	current_h = tello.get_height()
	prev_err = 0
	ud_pid = [0.2,0.1] #up-down pid constants
	t1 = time.time()
	while True:
		if (time.time()-t1) > 5:
			break
		current_h = tello.get_height()
		err = (desired_h - current_h)*3
		ud_vel = ud_pid[0]*err + ud_pid[1]*(err - prev_err)
		ud_vel = int(np.clip(ud_vel, -100, 100))
		prev_err = err
		tello.send_rc_control(0, 0, ud_vel, 0)
		time.sleep(0.05)
		print(f'hovering at height: {current_h} cms')
	print(f'exiting hover function')

def go_back(tello):
	print('going back...')
	tello.move_back(100)
	print('moving back complete!')
		
def call_mavic(tello):
	for i in range(4):
		print('aaja mavic bhai, bachale mujhe')
	
def land(tello):
	print('landing initiated....')
	tello.land()
	print('landing complete!')

def end_loop(tello, mode):
	cv2.destroyAllWindows()
	if mode != 0:
		tello.streamoff()
		tello.land() 
	print("done executing")

def keyboard_control(tello):
	while True:
		img = tello.get_frame_read().frame
		cv2.imshow("drone", img)
		key = cv2.waitKey(1) & 0xff
		if key == 27: # ESC
		    break
	    #(lr_vel, fb_vel, ud_vel, yaw_vel)
		elif key == ord('w'): #front 
		    tello.send_rc_control(0, 50, 0, 0)
		elif key == ord('s'): #back
		    tello.send_rc_control(0, -50, 0, 0)
		elif key == ord('a'): #left
		    tello.send_rc_control(-50, 0, 0, 0)
		elif key == ord('d'): #right
		     tello.send_rc_control(50, 0, 0, 0)
		elif key == ord('e'): #yaw right
		    tello.send_rc_control(0, 0, 0, 30)
		elif key == ord('q'): #yaw left
		    tello.send_rc_control(0, 0, 0, -30)
		elif key == ord('r'): #up
		    tello.send_rc_control(0, 0, 30, 0)
		elif key == ord('f'): #down
		    tello.send_rc_control(0, 0, -30, 0)
		elif key == ord('x'): #down
		    tello.send_rc_control(0, 0, 0, 0)
		else:
			continue
	cv2.destroyAllWindows()
	tello.streamoff()
	tello.land()

def dummy_control(tello):
	print('starting sequence of commands....')
	tello = tello_start()
	tello_first_takeoff(tello)
	for i in range(3):
		rotate_ccw(tello, i)
	lower_height(tello)
	go_back(tello)
	hover(tello)
	call_mavic(tello)
	land(tello)
	tello.streamoff()
	cv2.destroyAllWindows()
	print('code executed successfully!')

def find_person_logic(tello):
	high_h_ctr, low_h_ctr, go_back_status = 0,0,False
	tello_first_takeoff(tello)
	i = 0
	while True:
		i += 1
		image = tello.get_frame_read().frame
		cv2.imshow('image', image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if high_h_ctr < 6:
			rotate_ccw(tello, i, 60)
			high_h_ctr += 1
		if high_h_ctr >= 6:
			if low_h_ctr == 0:
				lower_height(tello)
				low_h_ctr += 1
				continue
			rotate_ccw(tello, i, 45)
			low_h_ctr += 1
			if low_h_ctr > 8:
				if go_back_status == False:
					go_back(tello)
					go_back_status = True
					continue
				rotate_ccw(tello, i, 45)
				low_h_ctr += 1
				if low_h_ctr >= 16:
					call_mavic(tello)
					land(tello)
					break
		continue
	tello.streamoff()
	cv2.destroyAllWindows()
	print(f'code executed successfully!, battery: {tello.get_battery()}')

def pose_based_control(tello, pose, pose_classify_model, bodypose, bodypose_array, i, image, out):
	#logic for bodypose-based drone control
	bodypose_array.append(bodypose)
	last_poses = bodypose_array[-8:-1]
	print(last_poses)
	special_poses = ['upright', 'start_stop', 'land_go_back', 'kneel', 'fall']
	for special_pose in special_poses:
		itr = 0
		for j in last_poses:
			if j == special_pose:
				itr += 1
		if itr > 5:
			print(itr, special_pose)
			if special_pose == 'upright':
				cv2.putText(image, 'SAFE', (700,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3, cv2.LINE_AA)
				
			if special_pose == 'start_stop':
			    bodypose_array, i = start_stop_hover(tello, pose, pose_classify_model, bodypose_array, out, i )
				
			if special_pose == 'land_go_back':
				land_upon_prompt(tello, out, i)
				
			if special_pose == 'kneel' or special_pose == 'fall':
				cv2.putText(image, 'FALL DETECTED', (600,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
	
	return bodypose_array, i


def start_stop_hover(tello, pose, pose_classify_model, bodypose_array, out = 'images', i = 0):
	print('hovering at height 120 cms....')
	for m in range(5):
		bodypose_array.append('placeholder')
	desired_h = 130
	current_h = tello.get_height()
	prev_err = 0
	ud_pid = [0.2,0.1] #up-down pid constants
	t1 = time.time()
	new_array = []
	special_poses = ['start_stop', 'kneel', 'fall']
	while True:
		current_h = tello.get_height()
		err = (desired_h - current_h)*3
		ud_vel = ud_pid[0]*err + ud_pid[1]*(err - prev_err)
		ud_vel = int(np.clip(ud_vel, -100, 100))
		prev_err = err
		tello.send_rc_control(0, 0, ud_vel, 0)
		time.sleep(0.05)
		print(f'hovering at height: {current_h} cms')
		
		#condition for break
		image = tello.get_frame_read().frame
		image, inf_time, bodypose = pose_detect(image, pose, pose_classify_model)
		cv2.putText(image, 'HOVER', (700,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA) #BGR
		cv2.imwrite(f'{out}/{i}.png', cv2.resize(image, (416, 416), interpolation=cv2.INTER_AREA))
		cv2.imshow('frame', image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
					
		new_array.append(bodypose)
		last_poses = new_array[-8:-1]
		exit_loop = False
		for special_pose in special_poses:
			itr2 = 0						 
			for k in last_poses:
				if k == special_pose:
					itr2 += 1
			if itr2 >= 5:
				if special_pose == 'start_stop':
					for u in range(5):
						bodypose_array.append('upright')
				exit_loop = True
				break
		if exit_loop:
			break
		i += 1
	print(f'exiting hover function')
	return bodypose_array, i
	

def land_upon_prompt(tello, out = 'images', i = 0):
	print('Landing initiated')
	while True:	
		image = tello.get_frame_read().frame
		cv2.putText(image, 'LAND DRONE!', (600,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
		cv2.imwrite(f'{out}/{i}.png', cv2.resize(image, (416, 416), interpolation=cv2.INTER_AREA))
		cv2.imshow('frame', image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		i += 1
		print(f'height: {tello.get_height()}')
		tello.send_rc_control(0, 0, -30, 0)
		time.sleep(0.3)
		if tello.get_height() < 35:
			break
	land(tello)
	tello.streamoff()
	cv2.destroyAllWindows()
	prv.video(out, '' , '0')
	sys.exit('terminating program')
		
if __name__ == '__main__':
	tello = tello_start()
	tello_first_takeoff(tello)
	land_upon_prompt(tello)

#	mode = 1
#	try: 
#		if mode == 0:
#			dummy_control(tello)
#		else:
#			find_person_logic(tello)
#	except KeyboardInterrupt:
#		end_loop(tello, mode)
	

	
	
	

