# Yolov7Wrapper.py
# A wrapper around Yolov7 to neatly encapsulate all the relevant inference settings in one place.

import os, sys, logging
from re import S
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../yolov7/seg"))
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords, strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks


class Yolov7Wrapper(pl.LightningModule):
    """A wrapper around Yolov7-seg. Relies on the presence of the yolov7 repository, with the u7 branch checked out
    (at the time of writing, only the u7 branch contains code for segmentation).
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.yolov7_chkpt_path = self.args.graphbins.get('yolov7_chkpt')
        # self.detector = torch.hub.load("./yolov7/seg/", model="custom", source="local", path=self.yolov7_chkpt_path)
        # self.detector = self.detector.model
        # device = select_device()

        # Size gets overwritten every epoch; this is just for warmup.
        self.imgsz = self.args[self.args.basic.dataset].dimensions_train
        self.detector = DetectMultiBackend(self.yolov7_chkpt_path, device=self.device, dnn=False, data=None, fp16=False)
        self.stride, self.class_names, pt = self.detector.stride, self.detector.names, self.detector.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.detector.warmup(imgsz=(1 if pt else self.args.basic.batch_size, 3, *self.imgsz))  # warmup

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    # ImageNet statistics
        

    def forward(self, image, original_image=None):
        """
        Args:
            image (torch.Tensor, Bx3xHxW): batch of images to be annotated
            original_image (optional torch.Tensor, Bx3xHxW): if images have been resized, original_image is batch of un-resized images.
        
        Returns:
            xywh, confs, masks, cls, names, annotated_images
                xywh (length B list of torch.Tensor, Nx4): centre coordinate (xy), width, and height of bbox for N instances. In pixels.
                confs (length B list of torch.Tensor, N): confidence values for N detections
                masks (length B list of torch.Tensor, NxHxW): pixelwise masks for each of N detections
                cls (length B list of torch.Tensor, N): class index for N detections)
                names (length B list of strings): Name of each class (in natural language).
                annotated_images (torch.Tensor, Bx3xHxW): batch of annotated images to be plotted.
        """
        xywh_list = []
        confs_list = []
        masks_list = []
        cls_list = []
        names_list = []
        annotated_image_list = []   # Will get concatenated together at the end along the batch dimension
        
        if original_image is None:
            original_image = image.clone()

        unnormed_image = image.clone() * torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
    
        # Because of how yolov7 works, images must be padded to become a multiple of the max stride of the model (32 by default).
        # if self.training:
        #     self.imgsz = self.args[self.args.basic.dataset].dimensions_train
        # else:
        #     self.imgsz = self.args[self.args.basic.dataset].dimensions_test
        # self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        
        # sz_diff = [self.imgsz[0] - unnormed_image.shape[2], self.imgsz[1] - unnormed_image.shape[3]]
        # padded_image = F.pad(unnormed_image, (0, sz_diff[1], 0, sz_diff[0]), value=0)   # F.pad dims are backwards - see docs
        padded_image = unnormed_image

        self.detector.eval()    # See yolo.py/ISegment
        # pred, out = self.detector(unnormed_image) if self.training else self.detector(unnormed_image, val=True)
        # pred, out = pred[0], pred[1]
        pred, out = self.detector(padded_image)
        if type(pred) in [list, tuple]:
            sys.exit("Error: detector is in training mode, so output is in the wrong format.")
            return xywh_list, masks_list, confs_list, cls_list, names_list, None
        else:
            proto = out[1]
            pred = non_max_suppression(
                prediction=pred,
                conf_thres=self.args.yolov7seg.conf_thres,
                iou_thres=self.args.yolov7seg.iou_thres,
                classes=None,
                agnostic=self.args.yolov7seg.agnostic_nms,
                max_det=self.args.yolov7seg.max_det,
                nm=32
            )
            for i, det in enumerate(pred):  # per image (len(pred) == B, even if pred[i] has nothing in it.)
                im0 = original_image[i].permute(1, 2, 0).contiguous()    # im0 is meant to be HWC, but original_image is BCHW.
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0.cpu().numpy(), line_width=2, example=None)
                
                if det.shape[0] > 0:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], padded_image.shape[2:], upsample=True)  # HWC

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(padded_image.shape[2:], det[:, :4], im0.shape).round()

                    # Mask plotting ----------------------------------------------------------------------------------------
                    mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                    im_masks = plot_masks(padded_image[i], masks, mcolors)  # padded_image with masks shape(imh,imw,3)
                    annotator.im = scale_masks(padded_image.shape[2:], im_masks, im0.shape)  # scale to original h, w

                    # Write results.
                    # xyxy: Nx[x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right (both in pixels)
                    # xywh: Nx[x, y, w, h] where xy=centre (in pixels), w=width, h=height.
                    xyxy = reversed(det[:, :4])
                    confs = reversed(det[:, 4:5]).squeeze(1)
                    cls = reversed(det[:, 5:6]).int().squeeze(1)
                    xywh = xyxy2xywh(xyxy)
                    names = []

                    for i, ci in enumerate(cls):
                        # Name list and bbox plotting
                        name = self.class_names[int(ci)]
                        names.append(name)
                        label = f'{name} {confs[i]:.2f}'
                        annotator.box_label(xyxy[i], label, color=colors(int(ci), True))
                else:
                    # To ensure there's always something returned for every case
                    xywh = None
                    masks = None
                    confs = None
                    cls = None
                    names = None

                xywh_list.append(xywh)
                masks_list.append(masks)
                confs_list.append(confs)
                cls_list.append(cls)
                names_list.append(names)

                annotated_image_list.append(annotator.im)  # Outside if statement to ensure it's always added

            annotated_images = np.stack(annotated_image_list, axis=0)
            annotated_images = torch.from_numpy(annotated_images).permute(0, 3, 1, 2).contiguous()
            return xywh_list, masks_list, confs_list, cls_list, names_list, annotated_images
