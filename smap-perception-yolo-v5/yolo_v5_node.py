#!/usr/bin/env python3

from std_srvs.srv import Empty
from std_msgs.msg import String
#from smap_classification_wrapper.smap_classification_wrapper import classification_wrapper
from smap_classification_wrapper.classification_wrapper import classification_wrapper,main
from smap_interfaces.msg import SmapData, SmapPrediction

#import cv2
#from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

import torch
import json

# YOLO_V5 functionalities adapted from https://github.com/ultralytics/yolov5/blob/master/models/common.py

import time
import torchvision
import numpy as np
import math

from smap_perception_yolo_v5.yolov5.detect import run

class yolo_v5(classification_wrapper):

    def __init__(self,detector_name='yolo_v5'):
        super().__init__(
            detector_name=detector_name,
            detector_type='object',
            detector_architecture='yolo_v5'
        )

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_file="/workspace/install/share/smap_perception_yolo_v5/yolov5/yolov5s.torchscript"
        self.imgsz=(1, 3, 640, 640)

        run()

        #self.br = CvBridge()
        
    def predict(self,msg):
        #msg.stamped_pose
        #msg.rgb_image
        #msg.pointcloud

        print('<Pred')
        

        #img=torch.from_numpy(img).to(self.model.device)
        #img=img.float()
        #img/=255
        #if len(img.shape) == 3:
        #    img = img[None]  # expand for batch dim
        #pred = self.model(img)
        #pred = non_max_suppression(
        #    pred,
        #    conf_thres=0.25,
        #    iou_thres=0.54,
        #    classes=None,
        #    agnostic_nms=False,
        #    max_det=1000
        #)

        #current_frame = self.br.imgmsg_to_cv2(msg.rgb_image)
        print(msg.rgb_image.height)
        print(msg.rgb_image.width)

        #cv2.imshow("camera", current_frame)
    
        #cv2.waitKey(0)
        

        resp_msg = SmapPrediction()
        resp_msg.module_id = 5
        self.publisher.publish(resp_msg)
        print('>Pred')

    def _load_model(self): # Return True when an error occurs
        if super()._load_model():  
            return True
        self.model_file = str(self.model_file)
        extra_files = {'config.txt': ''}  # model metadata
        self.model = torch.jit.load(self.model_file, _extra_files=extra_files, map_location=self.device)
        if extra_files['config.txt']:  # load metadata dict
            d = json.loads(extra_files['config.txt'],
                            object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                    for k, v in d.items()})
            self.stride, self.classes = int(d['stride']), d['names']

        self.get_logger().info("{} loaded.".format(self.model_file))
        return False
    
    def _model_warmup(self):
        self.get_logger().info('Model warming up...')
        img = torch.empty(*self.imgsz, dtype=torch.float, device=self.device)  # input
        self.model.forward(img)



if __name__ == '__main__':

    detector_args = {
        'name': 'yolo_v5'
    }
    main(detector_class=yolo_v5,detector_args=detector_args)