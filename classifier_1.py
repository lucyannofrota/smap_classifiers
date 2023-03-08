#!/usr/bin/env python3

from std_srvs.srv import Empty
from std_msgs.msg import String
#from smap_classification_wrapper.smap_classification_wrapper import classification_wrapper
from smap_classification_wrapper.classification_wrapper import classification_wrapper


class classifier_1(classification_wrapper):

    def __init__(self,node_name):
        super().__init__(node_name)
        