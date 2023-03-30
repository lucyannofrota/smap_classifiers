#!/usr/bin/env python3

# Node based on https://github.com/ultralytics/yolov5

from sensor_msgs.msg import Image
from smap_classification_wrapper.classification_wrapper import classification_wrapper,main
from smap_interfaces.msg import SmapData, SmapPrediction, SmapDetectionDebug

import cv2
import torch
import numpy as np

# ultralytics
from models.common import DetectMultiBackend
from utils.plots import colors
from utils.plots import Annotator, colors
from utils.augmentations import letterbox
from utils.general import Profile, check_img_size, non_max_suppression, scale_boxes

# TODO: Separar pre-processing. inference e nms

class yolo_v5(classification_wrapper):

    def __init__(self,detector_name='yolo_v5'):
        super().__init__(
            detector_name=detector_name,
            detector_type='object',
            detector_architecture='yolo_v5'
        )

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_file="/workspace/install/share/smap_perception_yolo_v5/weights/yolov5s.torchscript"
        self.model_description_file="/workspace/install/share/smap_perception_yolo_v5/data/coco128.yaml"
        self.imgsz=(640, 640)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.half=False

    def initialization(self):
        self.get_logger().debug("Initializing topics")
        self.subscription=self.create_subscription(SmapData, '/smap/sampler/data', self.predict, 10,callback_group= self._reentrant_cb_group)
        self.prediction=self.create_publisher(SmapPrediction, '/smap/perception/predictions', 10,callback_group= self._reentrant_cb_group)
        self.publisher_debug_image=self.create_publisher(Image, '/smap/perception/predictions/debug', 10,callback_group= self._reentrant_cb_group)
        return True

    def predict(self,msg):
        #msg.stamped_pose
        #msg.rgb_image
        #msg.pointcloud

        # pre-processing
        with self.pre_processing_tim:
            self._img_original = self._cv_bridge.imgmsg_to_cv2(msg.rgb_image, "bgr8")
            self._img_processed = letterbox(self._img_original, self.imgsz, stride=self.stride, auto=self.model.pt)[0]  # padded resize

            self._img_processed = self._img_processed.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            self._img_processed = np.ascontiguousarray(self._img_processed)  # contiguous

            self._img_processed = torch.from_numpy(self._img_processed).to(self.model.device)
            self._img_processed = self._img_processed.half() if self.model.fp16 else self._img_processed.float()  # uint8 to fp16/32
            self._img_processed /= 255  # 0 - 255 to 0.0 - 1.0
            if len(self._img_processed.shape) == 3:
                self._img_processed = self._img_processed[None]  # expand for batch dim


        # inference
        with self.inference_tim:
            pred = self.model(self._img_processed, augment=False)

        # nms
        with self.nms_tim:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Apply Classifier
        #print('Classify')
        #if self.classify:
        #    pred = apply_classifier(pred, self.modelc, img, pred)

        # Process detections
        with self.post_processing_tim:
            s=''
            for i, det in enumerate(pred):  # per image
                s += '%gx%g ' % self._img_processed.shape[2:]  # print string
                annotator = Annotator(self._img_original, line_width=3, example=str(self.names))
                if len(det):
                    # Rescale boxes from self._img_processed to self._img_original size
                    det[:, :4] = scale_boxes(self._img_processed.shape[2:], det[:, :4], self._img_original.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    # TODO: Create parameter
                    view_img=True
                    hide_labels=False
                    hide_conf=False
                    for *xyxy, conf, cls in reversed(det):
                        if view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                self._img_original = annotator.result()
                #view_img=False

        self.get_logger().debug(f"{s}{'' if len(det) else '(no detections), '}{self.inference_tim.t:.1f}ms")
        
        # Print results
        self.get_logger().debug(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms nms, %.1fms post-processing | per image at shape {(1, 3, *self.imgsz)}' % (self.pre_processing_tim.t, self.inference_tim.t, self.nms_tim.t, self.post_processing_tim.t))
        self.get_logger().debug('Total process time: {:0.1f}ms | Should be capable of: {:0.1f} fps'.format(self.get_callback_time(),1E3/self.get_callback_time()))
        self.get_logger().debug(f'Total process time: %.1fms' % self.get_callback_time())
        

        if view_img:
            self.publisher_debug_image.publish(self._cv_bridge.cv2_to_imgmsg(self._img_original))
        
        resp_msg = SmapPrediction()
        resp_msg.module_id = self.module_id
        self.prediction.publish(resp_msg)

    def load_model(self): # Return True when an error occurs
        if super().load_model():  
            return True
        
        # Load model
        self.model = DetectMultiBackend(weights=self.model_file,
                                   device=self.device,
                                   dnn=False,
                                   data=self.model_description_file,
                                   fp16=self.half
        )
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        return False
    
    def load_dataloader(self): # Return True when an error occurs
        super().load_dataloader()
        return False

    def model_warmup(self):
        super().model_warmup()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup

if __name__ == '__main__':

    detector_args = {
        'name': 'yolo_v5'
    }
    main(detector_class=yolo_v5,detector_args=detector_args)