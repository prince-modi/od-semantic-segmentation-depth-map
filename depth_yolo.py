#pseudo code
import iros_utils as iu
import extract_frames as ef
import cv2

i = 0
while True:
    frame = ef.fetch_frame(i)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    i += 1

# frame = iu.fetch_frame()
# bboxes = yolo(frame)
# depth = midas(frame)
# rev_arr = iu.relative_dist(bboxes, depth)

#rev_arr contains obj class, id, dist, bbox coord
