#source badri/.venv/bin/activate
#cd cv_tasks/tracking/darknet-master/
#sudo update-alternatives --config python3
#df -h (memory)

import cv2
import os
import time

def video(out, csv_path='', class_name='0'):
	#playing/record process
	print(out)
	mode = input('Play(1) or make(2) video or skip(3)?: ')

	if (int(mode) == 1):
		print('play')
		i = 0
		t1 = time.time()
		while True:
			i += 1
			if os.path.exists(f'{out}/{i}.jpg'):
				image = cv2.imread(f'{out}/{i}.jpg')
				cv2.imshow('bodypose', image)
				if cv2.waitKey(70) & 0xFF == ord('q'):
					break
				t1 = time.time()
			if (time.time() - t1) > 3:
				break

	elif (int(mode) == 2):
		print('make')
		frame_size = (cv2.imread(f'{out}/1.jpg').shape[:2])
		frame_size = (frame_size[1], frame_size[0])
		print(frame_size)
#		video = cv2.VideoWriter(f'{csv_path}/{class_name}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, frame_size)
#		print('Saving video at: ', f'{csv_path}/{class_name}.avi')
		video = cv2.VideoWriter(f'{out}/{0}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, frame_size)
		print('Saving video at: ', f'{out}/{0}.avi')
		j = 0
		for i in range(len(os.listdir(out))):
			try:
				if os.path.exists(f'{out}/{j}.jpg'):
					frame = cv2.imread(f'{out}/{j}.jpg')
					#cv2.imshow('frame', frame)
					cv2.waitKey(60)
					video.write(frame)
					j += 1
					print('wrote', f'{out}/{j}.jpg')
			except Exception as e:
				print(e)
				pass
		video.release()
		
	else:
		print('not playing or recording')
    
	cv2.destroyAllWindows()
	
	
if __name__ == '__main__':
	out = '/home/dream-nano6/cv_tasks/tracking/darknet-master/bodypose/inference/output/outdoor-testing/03-54-21'
#	out = '/media/dream-nano6/My Passport/tello/bodypose/inference/fall_outdoors'
	csv_path = '/home/dream-nano6/cv_tasks/tracking/darknet-master/bodypose'
	class_name = 'follow_me'
	video(out, csv_path, class_name)
