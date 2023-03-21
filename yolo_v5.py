#!/usr/bin/env python3

from std_srvs.srv import Empty
from std_msgs.msg import String
#from smap_classification_wrapper.smap_classification_wrapper import classification_wrapper
from smap_classification_wrapper.classification_wrapper import classification_wrapper
from smap_classification_wrapper.classification_wrapper import main
from smap_interfaces.msg import SmapData, SmapPrediction

class yolo_v5(classification_wrapper):

    def __init__(self,classifier_name='yolo_v5'):
        super().__init__(
            classifier_name=classifier_name,
            classifier_type='object',
            classifier_architecture='YOLO_V5'
        )
        
    def predict(self,data):
        msg = SmapPrediction()
        msg.classifier_id = 5
        self.publisher.publish(msg)


if __name__ == '__main__':

    classifier_args = {
        'name': 'yolo_v5'
    }
    main(classifier_class=yolo_v5,classifier_args=classifier_args)