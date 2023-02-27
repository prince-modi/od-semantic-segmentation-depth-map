import sys
sys.path.append("ByteTrack")
sys.path.append("ultralytics")
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from ultralytics import YOLO
from typing import List
import numpy as np
import cv2 as cv

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

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

class DetectTrack:
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


    def detect_track(self, frame, save=False, show=True):
        results = self.model(frame)
        self.frame = frame
    
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        tracks = self.byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=self.frame.shape,
            img_size=self.frame.shape
        )
        tracker_id = match_detections_with_tracks(
            detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        labels = [
            f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        self.line_counter.update(detections=detections)
        self.frame = self.box_annotator.annotate(
            frame=self.frame, detections=detections, labels=labels)
        self.line_annotator.annotate(frame=self.frame, line_counter=self.line_counter)

        if save or show:
            return results, self.frame
        else:
            return results, None