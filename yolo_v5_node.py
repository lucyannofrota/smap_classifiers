#!/usr/bin/env python3

# Node based on https://github.com/ultralytics/yolov5

from sensor_msgs.msg import Image
from smap_perception_wrapper.perception_wrapper import perception_wrapper, main
#from smap_perception_wrapper.perception_wrapper import main
#from smap_perception_wrapper.smap_perception_wrapper import perception_wrapper, main
from smap_interfaces.msg import SmapObject, SmapDetections

import torch
import numpy as np
import torchvision
import time

# ultralytics
from models.common import DetectMultiBackend
from utils.plots import colors
from utils.plots import Annotator, colors
from utils.general import check_img_size, scale_boxes, xywh2xyxy
from utils.metrics import box_iou


# TODO: Separar pre-processing. inference e nms

class yolo_v5(perception_wrapper):

    def __init__(self,detector_name='yolo_v5'):
        super().__init__(
            detector_name=detector_name,
            detector_type='object',
            detector_architecture='yolo_v5'
        )

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_file="/workspace/install/share/smap_yolo_v5/weights/yolov5s.engine"
        self.model_description_file="/workspace/install/share/smap_yolo_v5/data/coco128.yaml"
        self.imgsz=(640, 640)

        # NMS
        self.conf_thres=0.4  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.agnostic_nms=False  # class-agnostic NMS
        self.half=False

        self.counter = 0
        self.pre_vals = []
        self.inference_vals = []
        self.nms_vals = []
        self.post_vals = []

        # Checks
        assert 0 <= self.conf_thres <= 1, f'Invalid Confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= self.iou_thres <= 1, f'Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0'


        # TODO: Create parameter
        self.hide_labels=False
        self.hide_conf=False


    def initialization(self):
        if super().initialization():  
            # TODO: Debug parameter
            self.publisher_debug_image=self.create_publisher(Image, self.get_namespace()+'/perception/predictions/debug', 10,callback_group=self._reentrant_cb_group)
            return True
        return False

    def predict(self,msg):
        #msg.stamped_pose
        #msg.rgb_image
        #msg.pointcloud

        # pre-processing
        try:
            with self.pre_processing_tim:
                _img_original = self._cv_bridge.imgmsg_to_cv2(msg.rgb_image, "bgr8")
                _img_processed = self.letterbox(_img_original, self.imgsz, stride=self.stride, auto=self.model.pt)[0]  # padded resize

                _img_processed = _img_processed.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                _img_processed = np.ascontiguousarray(_img_processed)  # contiguous

                _img_processed = torch.from_numpy(_img_processed).to(self.model.device)
                _img_processed = _img_processed.half() if self.model.fp16 else _img_processed.float()  # uint8 to fp16/32
                _img_processed /= 255  # 0 - 255 to 0.0 - 1.0
                if len(_img_processed.shape) == 3:
                    _img_processed = _img_processed[None]  # expand for batch dim
        except (Exception, RuntimeError)  as e:
            self.get_logger().error("yolo_v5/predict/pre_processing")

        # inference
        with self.inference_tim:
            try:
                pred = self.model(_img_processed, augment=False)
            except (Exception, RuntimeError)  as e:
                self.get_logger().error("yolo_v5/predict/inference")

        # nms
        with self.nms_tim:
            try:
                pred = yolo_v5.non_max_suppression(pred, self.conf_thres, self.iou_thres, self.agnostic_nms, max_det=self.max_det)
            except (Exception, RuntimeError)  as e:
                self.get_logger().error("yolo_v5/predict/nms")
        # Apply Classifier
        #print('Classify')
        #if self.classify:
        #    pred = apply_classifier(pred, self.modelc, img, pred)

        # Process detections
        objects = []
        with self.post_processing_tim:
            s=''
            if self.get_logger().get_effective_level() == self.get_logger().get_effective_level().DEBUG:
                annotator = Annotator(_img_original, line_width=3, example=str(self.classes))

            for i, det in enumerate(pred):  # per image
                s += '%gx%g ' % _img_processed.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from _img_processed to _img_original size
                    det[:, :4] = scale_boxes(_img_processed.shape[2:], det[:, :4], _img_original.shape).round()
                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.classes[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for x1, y1, x2, y2, conf, cls, *probs in reversed(det):
                        xyxy = [x1, y1, x2, y2]
                        # *xyxy: [tensor(182., device='cuda:0'), tensor(209., device='cuda:0'), tensor(233., device='cuda:0'), tensor(233., device='cuda:0')]
                        # conf: tensor(0.56210, device='cuda:0')
                        # cls: tensor(64., device='cuda:0')
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.classes[c] if self.hide_conf else f'{self.classes[c]} {conf:.2f}')
                        obj = SmapObject()
                        obj.label = c
                        obj.bb_2d.keypoint_1 = [int(xyxy[0]),int(xyxy[1])]
                        obj.bb_2d.keypoint_2 = [int(xyxy[2]),int(xyxy[3])]
                        obj.confidence = float(conf)
                        obj.probability_distribution = [float(p) for p in probs]
                        objects.append(obj)

                        # Add bbox to image
                        if self.get_logger().get_effective_level() == self.get_logger().get_effective_level().DEBUG:
                            annotator.box_label(xyxy, label, color=colors(c, True))

                    # Stream results
                    if self.get_logger().get_effective_level() == self.get_logger().get_effective_level().DEBUG:
                        _img_original = annotator.result()

        self.get_logger().info(f"{s}{'' if len(det) else '(no detections), '}{self.inference_tim.t:.1f}ms",throttle_duration_sec=1)
        
        # Print results
        self.get_logger().debug(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms nms, %.1fms post-processing | per image at shape {(1, 3, *self.imgsz)}' % (self.pre_processing_tim.t, self.inference_tim.t, self.nms_tim.t, self.post_processing_tim.t))
        self.get_logger().debug('Total process time: {:0.1f}ms | Should be capable of: {:0.1f} fps'.format(self.get_callback_time(),1E3/self.get_callback_time()))
        self.get_logger().debug(f'Total process time: %.1fms' % self.get_callback_time())

        if self.get_logger().get_effective_level() == self.get_logger().get_effective_level().DEBUG:
            self.publisher_debug_image.publish(self._cv_bridge.cv2_to_imgmsg(_img_original))
    

        if self.get_logger().get_effective_level() == self.get_logger().get_effective_level().DEBUG:
            self.mean_spead_metrics(self.pre_processing_tim.t, self.inference_tim.t, self.nms_tim.t, self.post_processing_tim.t)
        
        # self.detections.publish(resp_msg)

        resp_msg = SmapDetections()
        resp_msg.objects = objects

        return resp_msg
    
    def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
        max_det=300,
        nm=0,  # number of masks
    ):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        # Checks
        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        # device = prediction.device
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        merge = False  # use merge-NMS

        t = time.time()
        # mi = 5 + nc  # mask start index
        # output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        output_probs = [torch.zeros((0, 6 + nm + nc), device=prediction.device)] * bs
        for xi, probs in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            probs = probs[xc[xi]]  # confidence

            # If none remain process next image
            if not probs.shape[0]:
                continue

            # Compute conf
            probs[:, 5:] *= probs[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = xywh2xyxy(probs[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            # mask = probs[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            # Softmax applied to class probabilities
            conf, j = probs[:, 5:].max(1, keepdim=True)
            # probs = torch.cat((box, conf, j.float(), mask, probs[:,5:mi]), 1)[conf.view(-1) > conf_thres]

            # a = torch.tensor([[2,2],[4,4]])

            # b = torch.tensor([4,6])
            # torch.transpose(b,0,1)

            # c = torch.div(a.transpose(0,1),b).transpose(0,1)

            # new_probs = (torch.div(probs[:,5:].transpose(0,1),probs[:,5:].sum(1))).transpose(0,1)
            # probs = torch.cat((box, conf, j.float(), new_probs), 1)[conf.view(-1) > conf_thres]
            probs = torch.cat((box, conf, j.float(), (torch.div(probs[:,5:].transpose(0,1),probs[:,5:].sum(1))).transpose(0,1)), 1)[conf.view(-1) > conf_thres]
            # probs = torch.cat((box, conf, j.float(), mask, torch.softmax(probs[:,5:mi],1)), 1)[conf.view(-1) > conf_thres]
            # x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Check shape
            # n = x.shape[0]  # number of boxes
            n = probs.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            # x_idxs = x[:, 4].argsort(descending=True)[:max_nms] # sort by confidence and remove excess boxes
            x_idxs = probs[:, 4].argsort(descending=True)[:max_nms] # sort by confidence and remove excess boxes
            # x = x[x_idxs]
            probs = probs[x_idxs]

            # Batched NMS
            # c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            c = probs[:, 5:6] * (0 if agnostic else max_wh)  # classes
            # boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            boxes, scores = probs[:, :4] + c, probs[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                probs[i, :4] = torch.mm(weights, probs[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                probs[i, :4] = torch.mm(weights, probs[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            # output[xi] = x[i]
            output_probs[xi] = probs[i]
            if (time.time() - t) > time_limit:
                # LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output_probs

    def mean_spead_metrics(self, pre_processing_tim, inference_tim, nms_tim, post_processing_tim):
        if len(self.pre_vals) >= 128:
            self.pre_vals=self.pre_vals[1:]
            self.inference_vals=self.inference_vals[1:]
            self.nms_vals=self.nms_vals[1:]
            self.post_vals=self.post_vals[1:]
        else:
            self.counter+=1

        self.pre_vals.append(pre_processing_tim)
        self.inference_vals.append(inference_tim)
        self.nms_vals.append(nms_tim)
        self.post_vals.append(post_processing_tim)

        ms=[
            sum(self.pre_vals)/len(self.pre_vals),
            sum(self.inference_vals)/len(self.pre_vals),
            sum(self.nms_vals)/len(self.pre_vals),
            sum(self.post_vals)/len(self.pre_vals)
        ]

        self.get_logger().warn("Mean Values [{}] | {:0.1f}ms pre-process, {:0.1f}ms inference, {:0.1f}ms nms, {:0.1f}ms post-processing | Total: {:0.1f}ms".format(
                self.counter,
                ms[0],
                ms[1],
                ms[2],
                ms[3],
                sum(ms)
            ),
            throttle_duration_sec=5
        )

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
        self.stride, self.classes, self.pt = self.model.stride, self.model.names, self.model.pt
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