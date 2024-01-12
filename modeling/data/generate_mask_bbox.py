import sys

import numpy as np
import pyclipper
from shapely.geometry import Polygon
import cv2
import torch

from modeling.data.seg_detector_representer import SegDetectorRepresenter

sys.path.append('/home/yinyin/salient_text')

class BorderMap():
    def __init__(self):
        self.shrink_ratio = 0.4
        self.thresh_min = 0.3
        self.thresh_max = 0.7
        
    def __call__(self, image, boxes):

        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        for i in range(len(boxes)):
            self.draw_border_map(boxes[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min

        return canvas,mask
        
    def draw_border_map(self,polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.cal_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def cal_distance(self,xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result

    def extend_line(self,point_1, point_2, result):
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2
        
class LabelGeneration():
    
    def __init__(self):
        
        self.shrink_ratio = 0.4
        self.dilate_ratio = 1

        
    # GT txt file to array
    def get_annotations(self,gt_file):
        bounding_boxes = []
        with open(gt_file, 'r',encoding='utf-8-sig') as file:
            for line in file:
                # Split the line by commas and remove leading/trailing whitespace
                coordinates = line.strip().split(',')
                
                # Convert the coordinates to integers
                #coordinates = [int(coord) if coord != 'null' else None for coord in coordinates]
                #coordinates = [x for x in coordinates if x is not None]
                coordinates = coordinates[:8]
                coordinates = [int(value) for value in coordinates]
                # Append the coordinates to the list
                bounding_boxes.append(coordinates)
        bounding_boxes = np.array(bounding_boxes).reshape(-1, 4, 2)

        return bounding_boxes
    
    
    # GT txt file to array
    def get_annotations_icdar15(self,gt_file, remove_donotcare):
        bounding_boxes = []
        with open(gt_file, 'r',encoding='utf-8-sig') as file:
            for line in file:
                # Split the line by commas and remove leading/trailing whitespace
                coordinates = line.strip().split(',')
                
                # Convert the coordinates to integers
                if coordinates[8]!="###" or not remove_donotcare:
                    coordinates = coordinates[:8]
                    coordinates = [int(value) for value in coordinates]

                    # Append the coordinates to the list
                    bounding_boxes.append(coordinates)

        bounding_boxes = np.array(bounding_boxes).reshape(-1, 4, 2)

        return bounding_boxes
    
    def dilate_polygon(self,polygon, dilate_ratio):
        """
        Dilates a single polygon by the specified dilate ratio.
        Returns the dilated polygon.
        """
        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(dilate_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        return padded_polygon

    def dilate_map(self,mask):
        """
        Dilates a binary segmentation mask using the Vatti clipping algorithm.
        Returns the dilated mask.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dilated_mask = np.zeros_like(mask)

        for contour in contours:
            exterior_coords = [tuple(point[0]) for point in contour]
            dilated_polygon = self.dilate_polygon(exterior_coords, self.dilate_ratio)

            # Fill the dilated polygon in the mask
            cv2.fillPoly(dilated_mask, [dilated_polygon.astype(np.int32)], 1.0)

        return dilated_mask
    
    def shrink_polygon(self,polygon, shrink_ratio):
        """
        Shrinks a single polygon by the specified shrink ratio.
        Returns the shrunken polygon.
        """
        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrunken_polygon = np.array(padding.Execute(-distance)[0])  # Note the negative distance for shrinking
        return shrunken_polygon

    def shrunk_map(self,mask):
        """
        Shrinks a binary segmentation mask using the Vatti clipping algorithm.
        Returns the shrunken mask.
        """
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shrunken_mask = np.zeros_like(mask).astype(np.int32)

        for contour in contours:
            exterior_coords = [tuple(point[0]) for point in contour]
            shrunken_polygon = self.shrink_polygon(exterior_coords, self.shrink_ratio)

            # Fill the shrunken polygon in the mask
            cv2.fillPoly(shrunken_mask, [shrunken_polygon.astype(np.int32)], 1.0)

        return shrunken_mask

    def generate_border_map(self, image, boxes):
        border = BorderMap()

        return border(image, boxes)
    
    # GT boxes to seg mask w/o shrunk/dilate
    def box2mask(self, image, boxes):
        #height, width = image.shape[:2]
        width, height = image.size
        seg = np.zeros((height, width), dtype=np.float32)
        
        for box_index in range(len(boxes)):
            seg =  cv2.fillPoly(seg, pts=[boxes[box_index]], color=(255, 0, 0))
            
        return seg
        

    def mask2box(self,image,mask,box_mode="quad"):
    
        #height, width = image.shape[:2]
        width, height = image.size

        seg_detector = SegDetectorRepresenter()
        if box_mode=="quad":
            boxes_batch, scores_batch = seg_detector.boxes_from_bitmap(torch.tensor(mask),torch.tensor(mask),height,width)
            
        elif  box_mode=="poly":
            boxes_batch, scores_batch = seg_detector.polygons_from_bitmap(torch.tensor(mask),torch.tensor(mask),height,width)
            
        else:
            raise ValueError("Invalid box_mode. Supported values are 'quad' and 'poly'")
    
        return boxes_batch, scores_batch
    
    def draw_box(self,image,boxes):
        image = np.array(image) # pil to opencv img

        out = image.copy()
        for i in range(len(boxes)):

            pts = np.array(boxes[i], np.int32)
            pts = pts.reshape((-1,1,2))
            out = cv2.polylines(out,[pts],True,(0,255,255),2)
        return out 