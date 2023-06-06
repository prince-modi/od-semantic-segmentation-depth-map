import cv2 
import os
import time

out = "/home/sumanraj/IROS_official/finale2/edge/free_road"
folder = "main"
duration = 10 #in seconds
frames = os.listdir(f"{out}/{folder}")
frame_size = (cv2.imread(f"{out}/{folder}/{frames[0]}").shape[:2])
frame_size = (frame_size[1], frame_size[0])
fps = int(len(frames)/duration)
video = cv2.VideoWriter(f"{out}/free_road_{folder}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 15, frame_size) #(1280, 720))

print(len(frames), frame_size, fps)
start = time.time()
for i in range(0, 250):
	try:
		frame = cv2.imread(f"{out}/{folder}/{i}.jpg")
		if folder == "without_mask":
			frame = cv2.imread(f"{out}/{folder}/{i}.png")
		# frame = cv2.resize(frame, (960, 720), interpolation = cv2.INTER_AREA)
		cv2.imshow('frame', frame)
		cv2.waitKey(1)
		video.write(frame)
		print(f'wrote frame: {out}/{i}.jpg')
	except:
		continue
print('total time:', time.time()-start)
print('time/frame:', (time.time()-start)/len(frames))
	
cv2.destroyAllWindows() 
video.release()  # releasing the video generated