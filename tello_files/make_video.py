import cv2 
import os
import time

out = "/home/sumanraj/IROS_official/finale2/partition/car_simple"
folder = "main"
frames = os.listdir(f"{out}/{folder}")
print(len(frames))
frame_size = (cv2.imread(frames[0]).shape[:2])
frame_size = (frame_size[1], frame_size[0])
print(frame_size)
video = cv2.VideoWriter(f"{out}/car_simple_original.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, frame_size)

start = time.time()
for i in range(50,350):
	try:
		frame = cv2.imread(f"{out}/{folder}/{i}.jpg")
		frame = cv2.resize(frame, (960, 720), interpolation = cv2.INTER_AREA)
	# cv2.imshow('frame', frame)
		cv2.waitKey(70)
		video.write(frame)
		print(f'wrote frame: {out}/{i}.jpg')
	except:
		continue
print('total time:', time.time()-start)
print('time/frame:', (time.time()-start)/len(frames))
	
cv2.destroyAllWindows() 
video.release()  # releasing the video generated


