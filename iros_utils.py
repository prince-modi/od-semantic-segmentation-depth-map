import numpy as np

def depth_pixels_bbox(bbox=None, depth=None):
    arr = np.random.randint(10, size=(10,10))
    xyxy = (3,3,6,7)
    bbox_pxls = arr[xyxy[0]:xyxy[2], xyxy[1]:xyxy[3]]
    median_dpt = np.median(bbox_pxls)
    print(bbox_pxls, np.sort(bbox_pxls.flatten()))
    print(median_dpt)
    print(np.where(bbox_pxls == median_dpt))

if __name__=="__main__":
    depth_pixels_bbox()

