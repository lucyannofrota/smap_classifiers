#!/usr/bin/env python3

from std_srvs.srv import Empty
from std_msgs.msg import String
#from smap_classification_wrapper.smap_classification_wrapper import classification_wrapper
from smap_classification_wrapper.classification_wrapper import classification_wrapper,main
from smap_interfaces.msg import SmapData, SmapPrediction, SmapDetectionDebug

#import cv2
#from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

import torch
import json

# YOLO_V5 functionalities adapted from https://github.com/ultralytics/yolov5/blob/master/models/common.py


# ultralytics
from detect import run
from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, apply_classifier)


from utils.plots import colors


from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

import numpy as np

# OpenCV
from cv_bridge import CvBridge, CvBridgeError
import cv2

import torchvision

from utils.plots import Annotator, colors, save_one_box

bridge = CvBridge()

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2
    else:  # np.array
        boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
        boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
        boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
        boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

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
        #self.imgsz=(896, 512)
        self.imgsz=(640, 640)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS

        

        #run(
        #    weights=self.model_file
        #)

        #self.br = CvBridge()
        self.half=False
        self.first_shot_=False
        torch.set_printoptions(profile="full")
        
    def initialization(self):
        self.get_logger().info("Initializing topics [Debug]")
        self.subscription=self.create_subscription(SmapData, '/smap/sampler/data', self.predict, 10,callback_group= self.reentrant_cb_group)
        self.publisher=self.create_publisher(SmapPrediction, '/smap/perception/predictions', 10,callback_group= self.reentrant_cb_group)
        self.publisher_debug=self.create_publisher(SmapDetectionDebug, '/smap/perception/predictions/debug', 10,callback_group= self.reentrant_cb_group)
        return True
    
    def first_shot(self,img):
        if not self.first_shot_:
            self.first_shot_ = not self.first_shot_
            print('Imshow')
            cv2.imwrite("/workspace/src/smap/smap_perception_yolo_v5/savedImage.jpg", img)
            cv2.imshow('img',img)
            #cv2.waitKey(0)
            print('Imshowc')


    def predict(self,msg):
        if self.first_shot_:
            return
        #self.first_shot_ = True
        #msg.stamped_pose
        #msg.rgb_image
        #msg.pointcloud

        print('<Pred')

        #img = bridge.imgmsg_to_cv2(msg.rgb_image)

        # Convert
        print('Resize')
        dt = (Profile(), Profile(), Profile(), Profile())
        #img = bridge.imgmsg_to_cv2(msg.rgb_image, "bgr8")
        img = cv2.imread('/workspace/src/smap/smap_perception_yolo_v5/weights/savedImage.jpg')
        #self.first_shot(img)
        
        with dt[0]:
            img_torch = letterbox(img, self.imgsz, stride=self.stride, auto=self.model.pt)[0]  # padded resize
        #im=img_torch
        #print("-----------------------")
        #print("-----------------------")
        #print("-----------------------")
        #print("-----------------------")
        #print("-----------------------")
        #print(im)
        #print(im.shape)
        #print(im.dtype)
        #print(self.imgsz)
        #print(self.stride)
        #print("-----------------------")
        #print("-----------------------")
        #print("-----------------------")
        #print("-----------------------")
        #print("-----------------------")
            img_torch = img_torch.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img_torch = np.ascontiguousarray(img_torch)  # contiguous

            print('Convert')
            img_torch = torch.from_numpy(img_torch).to(self.model.device)
            img_torch = img_torch.half() if self.model.fp16 else img_torch.float()  # uint8 to fp16/32
            img_torch /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img_torch.shape) == 3:
                img_torch = img_torch[None]  # expand for batch dim


        #s = np.stack([letterbox(x, self.imgsz, stride=self.stride)[0].shape for x in img], 0)  # shapes
        #self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        #if not self.rect:
        #    print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

        #print(img.shape)
        #img = torchvision.transforms.Resize(640)(img)
        #print(img.shape)
        #torch.Size([1, 3, 384, 640])
        # Inference
        with dt[1]:
            print('Inference')
            # TODO: Debug img_torch
            pred = self.model(img_torch, augment=False)
            #pred = self.model.model(img,
            #             visualize=increment_path('/workspace/src/smap/smap_perception_yolo_v5' / 'features', mkdir=True) if self.visualize else False)[0]

        # Apply NMS
        with dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Apply Classifier
        #print('Classify')
        #if self.classify:
        #    pred = apply_classifier(pred, self.modelc, img, pred)

        # Process detections
        with dt[3]:
            im=img_torch
            im0=img
            print('Predictions')
            # Process predictions
            s=''
            for i, det in enumerate(pred):  # per image
                print('Pred i:{}, det:{}'.format(i,det))
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0  # for save_crop
                annotator = Annotator(im0, line_width=3, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

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
                im0 = annotator.result()
                view_img=False
                if view_img:
                    cv2.imshow('YOLO_V5', im0)
                    cv2.waitKey(0)  # 1 millisecond

        self.get_logger().info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        
        # Print results
        t = tuple(x.t * 1E3 for x in dt)  # speeds per image
        self.get_logger().info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms Results | per image at shape {(1, 3, *self.imgsz)}' % t)
        tt=0
        print('tt')
        for x in dt:
            tt+=x.t
        print('tt sum')
        self.get_logger().info('Total process time: {:0.1f}ms | FPS: {:0.1f}'.format(tt*1E3,1/tt))
        self.get_logger().info(f'Total process time: %.1fms' % tt)

                





    #         if webcam:  # batch_size >= 1
    #             p, im0, frame = path[i], im0s[i].copy(), dataset.count
    #             s += f'{i}: '
    #         else:
    #             p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

    #         p = Path(p)  # to Path
    #         save_path = str(save_dir / p.name)  # im.jpg
    #         txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
    #         s += '%gx%g ' % im.shape[2:]  # print string
    #         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #         imc = im0.copy() if save_crop else im0  # for save_crop
    #         annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    #         if len(det):
    #             # Rescale boxes from img_size to im0 size
    #             det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

    #             # Print results
    #             for c in det[:, 5].unique():
    #                 n = (det[:, 5] == c).sum()  # detections per class
    #                 s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

    #             # Write results
    #             for *xyxy, conf, cls in reversed(det):
    #                 if save_txt:  # Write to file
    #                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    #                     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
    #                     with open(f'{txt_path}.txt', 'a') as f:
    #                         f.write(('%g ' * len(line)).rstrip() % line + '\n')

    #                 if save_img or save_crop or view_img:  # Add bbox to image
    #                     c = int(cls)  # integer class
    #                     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
    #                     annotator.box_label(xyxy, label, color=colors(c, True))
    #                 if save_crop:
    #                     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

    #         # Stream results
    #         im0 = annotator.result()
    #         if view_img:
    #             if platform.system() == 'Linux' and p not in windows:
    #                 windows.append(p)
    #                 cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    #                 cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
    #             cv2.imshow(str(p), im0)
    #             cv2.waitKey(1)  # 1 millisecond

    #         # Save results (image with detections)
    #         if save_img:
    #             if dataset.mode == 'image':
    #                 cv2.imwrite(save_path, im0)
    #             else:  # 'video' or 'stream'
    #                 if vid_path[i] != save_path:  # new video
    #                     vid_path[i] = save_path
    #                     if isinstance(vid_writer[i], cv2.VideoWriter):
    #                         vid_writer[i].release()  # release previous video writer
    #                     if vid_cap:  # video
    #                         fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #                         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                     else:  # stream
    #                         fps, w, h = 30, im0.shape[1], im0.shape[0]
    #                     save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
    #                     vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #                 vid_writer[i].write(im0)

    #     # Print time (inference-only)
    #     LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)












        #for i, det in enumerate(pred):  # detections per image
        #    s = f'{i}: '
        #    s += '%gx%g ' % img.shape[2:]  # print string
#
        #    if len(det):
        #        # Rescale boxes from img_size to im0 size
        #        det[:, :4] = scale_coords(img_torch.shape[2:], det[:, :4], img.shape).round()
#
        #        # Print results
        #        for c in det[:, -1].unique():
        #            n = (det[:, -1] == c).sum()  # detections per class
        #            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
        #        
        #        for *xyxy, conf, cls in reversed(det):
        #            c = int(cls)  # integer class
        #            label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
        #            plot_one_box(xyxy, img, label=label, color=colors(c, True), line_thickness=self.line_thickness)

        print('Ok')

        #print('Imshow')
        #cv2.imshow("Preview", img)
        #cv2.waitKey(0)    
        

        resp_msg = SmapDetectionDebug()
        resp_msg.module_id = 5
        resp_msg.rgb_image = msg.rgb_image
        self.publisher_debug.publish(resp_msg)
        print(msg.rgb_image.height)
        print(msg.rgb_image.width)
        print('>Pred')

    def _load_model(self): # Return True when an error occurs
        if super()._load_model():  
            return True
        #self.model_file = str(self.model_file)
        #extra_files = {'config.txt': ''}  # model metadata
        #self.model = torch.jit.load(self.model_file, _extra_files=extra_files, map_location=self.device)
        #if extra_files['config.txt']:  # load metadata dict
        #    d = json.loads(extra_files['config.txt'],
        #                    object_hook=lambda d: {int(k) if k.isdigit() else k: v
        #                                            for k, v in d.items()})
        #    self.stride, self.classes = int(d['stride']), d['names']

        # Load model
        self.model = DetectMultiBackend(weights=self.model_file,
                                   device=self.device,
                                   dnn=False,
                                   data=self.model_description_file,
                                   fp16=self.half
        )
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        self.get_logger().info("{} loaded.".format(self.model_file))
        return False
    
    def _dataloader(self): # Return True when an error occurs
        super()._dataloader()
        self.get_logger().info('Dataloader initialization complete.')
        return False

    def _model_warmup(self):
        super()._model_warmup()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
        self.get_logger().info('Warm up complete.')



if __name__ == '__main__':

    detector_args = {
        'name': 'yolo_v5'
    }
    main(detector_class=yolo_v5,detector_args=detector_args)