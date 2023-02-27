import sys
import os
sys.path.append("ByteTrack")
sys.path.append("ultralytics")
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from ultralytics import YOLO
import ultralytics
#import torch
from typing import List
import numpy as np
import cv2 as cv
import time



# ----------------------------------------------------------------------------
# Install ByteTrack


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


# ----------------------------------------------------------------------------
# Install Roboflow Supervision


# ----------------------------------------------------------------------------
# Tracking utils

# converts Detections into format that can be consumed by match_detections_with_tracks function

def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

# converts List[STrack] into format that can be consumed by match_detections_with_tracks function


def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

# matches our bounding boxes with predictions


def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


# ----------------------------------------------------------------------------
# Load pre-trained YOLOv8 model
# settings



class Detect_track:
    MODEL = ""
    mode = ""
    def __init__(self, mode):
        self.LINE_START = Point(50, 1500)
        self.LINE_END = Point(3840-50, 1500)
        
        # create BYTETracker instance
        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        # create LineCounter instance
        self.line_counter = LineCounter(start=self.LINE_START, end=self.LINE_END)
        # create instance of BoxAnnotator and LineCounterAnnotator
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=2)
        self.line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=2)

        self.mode = mode
        if self.mode=="yolo":
            MODEL = "/home/sumanraj/IROS_dummy/yolo+bytetrack/yolov8x.pt" 
            self.model = YOLO(MODEL)
            # self.model.fuse()

        if self.mode=="vest":
            MODEL = "/home/sumanraj/IROS_dummy/yolo+bytetrack/runs/detect/train1/weights/last.pt"
            self.model = YOLO(MODEL)
            # self.model.fuse()

        self.CLASS_NAMES_DICT = self.model.model.names
        # print(self.mode, self.CLASS_NAMES_DICT)
        # class_ids of interest - car, motorcycle, bus and truck
        # self.CLASS_ID = [i for i in range(0,80)]


    def detect_track(self, frame, itr, save=False, show=True):
        t1 = time.time()
        self.results = self.model(frame)
        t2 = time.time()
        detections = Detections(
            xyxy=self.results[0].boxes.xyxy.cpu().numpy(),
            confidence=self.results[0].boxes.conf.cpu().numpy(),
            class_id=self.results[0].boxes.cls.cpu().numpy().astype(int)
        )

        # tracking detections
        tracks = self.byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(
            detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)

        # filtering out detections without trackers
        mask = np.array(
            [tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # format custom labels
        self.labels = [
            f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        # updating line counter
        self.line_counter.update(detections=detections)

        # annotate and display frame
        frame = self.box_annotator.annotate(
            frame=frame, detections=detections, labels=self.labels)
        self.line_annotator.annotate(frame=frame, line_counter=self.line_counter)

        if save:
            # print(f'output/{self.mode}/{itr}.jpg')
            cv.imwrite(f'output/{self.mode}/{itr}.jpg', frame)
        
        if show:
            out = cv.resize(frame, (960, 720), interpolation = cv.INTER_AREA)
            cv.imshow('track_image', frame)
            cv.waitKey(1)
             
        t3 = time.time()
        # print(f'detect_loop: {t3-t1}, inferencing: {t2-t1}, overhead: {t3-t2}')

        return self.results



    
if __name__=="__main__":
    obj = Detect_track("yolo")
    i = 0
    while True:
        frame = cv.imread(f"frames/{i}.jpg")
        obj.detect_track(frame, i)
        i += 1