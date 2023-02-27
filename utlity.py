import cv2

def save_show(frame, itr, save = True, show = False):
    try:
        if save:
            cv2.imwrite(f'output/{itr}.jpg',frame)
        elif show:
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
        elif save and show:
            cv2.imwrite(f'output/{itr}.jpg',frame)
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
    finally:
        frame = None