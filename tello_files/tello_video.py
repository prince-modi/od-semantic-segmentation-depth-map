import cv2
import time
import play_record_video as prv
import datetime
import os
from djitellopy import Tello

def make_output_folder(location="test"):
    #modifying inference output folder
    out = '/home/sumanraj/IROS_official/tello_videos'
    out = os.path.join(out, location, datetime.datetime.now().strftime('%H-%M-%S'))
    print(f'created {out}')
    os.makedirs(out)
    print('output folder created')
    return out

def tello_start():
    tello = Tello()
    # tello.connect()
    # print('battery', tello.get_battery())
    # tello.streamon()
    # frame = tello.get_frame_read().frame
    return tello

def record_frames():
    tello = tello_start()
    out = make_output_folder(input('Input folder name: '))
    
    try:
        i=0
        while True:
            t1 = time.time()
            try:
                # image = tello.get_frame_read().frame
                image = cv2.imread(f"/home/sumanraj/IROS_official/frames/{i}.jpg")
                cv2.imwrite(f"{out}/{i}.jpg", image)
                cv2.imshow("frame", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
                i += 1
            except Exception as e:
                print('frame not captured', e)
                break

    except KeyboardInterrupt:
        cv2.destroyAllWindows()

    prv.video(out)    

if __name__=="__main__":
    record_frames()