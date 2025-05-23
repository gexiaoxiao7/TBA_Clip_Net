import numpy as np
import torch.nn as nn
from ultralytics import YOLO

import json
import cv2

# Load the track id from the file
class TeacherDetection(nn.Module):
    def __init__(self, yolo_model):
        super(TeacherDetection, self).__init__()
        self.model = YOLO(yolo_model)

    def forward(self, image):
        results = self.model(image)
        boxes = np.array(results[0].to('cpu').boxes.data)
        # track_ids = results[0].boxes.id.int().cpu().tolist()

        # Filter out boxes that are not 'person'
        person_boxes = [box for box in boxes if box[5] == 0]

        if not person_boxes:
            return image

        # Find the 'person' box with the smallest y-coordinate
        largest_box = min(person_boxes, key=lambda box: box[1])

        # Check if the top boundary of the largest box is in the upper half of the image
        if largest_box[1] > image.shape[0] / 2:
            return image

        # Find other 'person' boxes that intersect with the largest box
        intersecting_boxes = [box for box in boxes if self._intersect(box, largest_box)]

        # Combine all boxes into a large box
        # large_box = self._combine_boxes([largest_box] + intersecting_boxes)
        large_box = self._combine_boxes([largest_box])

        # Crop the image according to the large box
        cropped_image = image[int(large_box[1]):int(large_box[3]), int(large_box[0]):int(large_box[2])]

        return cropped_image

    @staticmethod
    def _intersect(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return x1 < x2 and y1 < y2

    @staticmethod
    def _combine_boxes(boxes):
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        return [x1, y1, x2, y2]