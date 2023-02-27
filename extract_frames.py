from moviepy.editor import VideoFileClip
import numpy as np
import os
from datetime import timedelta
import shutil
import argparse
import cv2 as cv
import time

SAVING_FRAMES_PER_SECOND = 10



def main(video_file, filename, extraction=False):
    # load the video clip
    video_clip = VideoFileClip(video_file)
    # make a folder by the name of the video file
    filename, _ = os.path.splitext(video_file) #not needed as of now

    if extraction == True:
        if os.path.isdir(filename):
            shutil.rmtree(filename)

        if not os.path.isdir(filename):
            os.mkdir(filename)

        # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
        saving_frames_per_second = min(
            video_clip.fps, SAVING_FRAMES_PER_SECOND)

        # if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
        step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / \
            saving_frames_per_second

        # iterate over each possible frame
        i = 0
        for current_duration in np.arange(0, video_clip.duration, step):
            # format the file name and save it
            frame_filename = os.path.join(filename, f"{i}.jpg")
            print(frame_filename)
            # save the frame with the current duration
            video_clip.save_frame(frame_filename, current_duration)
            i += 1

    return filename


def display(video_folder="output"):
    i = 0
    while True:
        input_path = f"{video_folder}/{i}.jpg"
        try: 
            try:
                frame = cv.imread(input_path)
                i += 1
            except Exception as e:
                print(f"frame not found: {input_path}")
                continue
            cv.imshow("frame", frame)
            if cv.waitKey(30) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break
        except KeyboardInterrupt as e:
            break
    cv.destroyAllWindows()
    print('done')

def fetch_frame(itr=None, input="frames"):
    input_path = f"{input}/{itr}.jpg"
    return cv.imread(input_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='sample_videos/footpath_edited.mp4',
                        help='video chalane ke liye')
    parser.add_argument('--display', default='False',
                        help='show frames')
    parser.add_argument('--output', default='extracted',
                        help='output folder for extracted frames')
    args = parser.parse_args()
                        
    video_file = args.source
    video_folder = args.output
    try: 
        video_folder = main(video_file, video_folder, True)
    except KeyboardInterrupt as e:
        pass
    print(video_folder)
    if args.display:
        display(video_folder)
