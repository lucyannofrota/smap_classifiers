#!/usr/bin/env python3

from std_srvs.srv import Empty
from std_msgs.msg import String
#from smap_classification_wrapper.smap_classification_wrapper import classification_wrapper
from smap_classification_wrapper.classification_wrapper import classification_wrapper,main
from smap_interfaces.msg import SmapData, SmapPrediction

import cv2


import torch
import json





class yolo_v5(classification_wrapper):

    def __init__(self,classifier_name='yolo_v5'):
        super().__init__(
            classifier_name=classifier_name,
            classifier_type='object',
            classifier_architecture='yolo_v5'
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def predict(self,data):
        msg = SmapPrediction()
        msg.classifier_id = 5
        self.publisher.publish(msg)

    def _load_model(self):
        super(yolo_v5, self)._load_model()
        self.model_file=0
        model_file = str(self.model_file)
        extra_files = {'config.txt': ''}  # model metadata
        self.model = torch.jit.load(model_file, _extra_files=extra_files, map_location=self.device)
        #model.half() if fp16 else model.float()
        if extra_files['config.txt']:  # load metadata dict
            d = json.loads(extra_files['config.txt'],
                            object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                    for k, v in d.items()})
            self.stride, self.classes = int(d['stride']), d['names']

        self.get_logger().warning("yolo_load_model")
        pass

if __name__ == '__main__':

    classifier_args = {
        'name': 'yolo_v5'
    }
    main(classifier_class=yolo_v5,classifier_args=classifier_args)