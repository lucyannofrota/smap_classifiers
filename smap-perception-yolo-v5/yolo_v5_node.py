#!/usr/bin/env python3

from std_srvs.srv import Empty
from std_msgs.msg import String
#from smap_classification_wrapper.smap_classification_wrapper import classification_wrapper
from smap_classification_wrapper.classification_wrapper import classification_wrapper,main
from smap_interfaces.msg import SmapData, SmapPrediction

import cv2


import torch
import json



import os

class yolo_v5(classification_wrapper):

    def __init__(self,detector_name='yolo_v5'):
        super().__init__(
            detector_name=detector_name,
            detector_type='object',
            detector_architecture='yolo_v5'
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_file = "../yolov5/yolov5s.torchscript"
        
    def predict(self,data):
        msg = SmapPrediction()
        msg.detector_id = 5
        self.publisher.publish(msg)

    def _load_model(self): # Return True when an error occurs
        if super()._load_model():  
            return True
        print('model_file')
        print(self.model_file)
        self.model_file = str(self.model_file)
        extra_files = {'config.txt': ''}  # model metadata
        print('pre')
        self.model = torch.jit.load(self.model_file, _extra_files=extra_files, map_location=self.device)
        print(self.model)
        #model.half() if fp16 else model.float()
        if extra_files['config.txt']:  # load metadata dict
            d = json.loads(extra_files['config.txt'],
                            object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                    for k, v in d.items()})
            self.stride, self.classes = int(d['stride']), d['names']

        self.get_logger().warning("yolo_load_model")
        return False


def _load_model(self): # Return True when an error occurs

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model_file = "yolov5/yolov5s.torchscript"
    print('model_file')
    print(self.model_file)
    self.model_file = str(self.model_file)
    
    extra_files = {'config.txt': ''}  # model metadata
    print('pre')
    self.model = torch.jit.load(self.model_file, _extra_files=extra_files, map_location=self.device)
    print(self.model)
    #model.half() if fp16 else model.float()
    if extra_files['config.txt']:  # load metadata dict
        d = json.loads(extra_files['config.txt'],
                        object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                for k, v in d.items()})
        self.stride, self.classes = int(d['stride']), d['names']

    self.get_logger().warning("yolo_load_model")
    return False

if __name__ == '__main__':

    detector_args = {
        'name': 'yolo_v5'
    }
    main(detector_class=yolo_v5,detector_args=detector_args)