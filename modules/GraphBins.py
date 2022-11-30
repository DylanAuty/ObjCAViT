# GraphBins.py
# Implements the GraphBins model as an nn.Module.

import os, sys, logging
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from nltk.corpus import wordnet as wn

from modules.DenseFeatureExtractor import DenseFeatureExtractor
from modules.ObjCAViT import ObjCAViT
from modules.ObjectLanguageStrategy import ObjectLanguageStrategy
from modules.LanguageEmbeddingWrapper import LanguageEmbeddingWrapper
from modules.Yolov7Wrapper import Yolov7Wrapper


class GraphBins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.logger = logging.getLogger(__name__)

        self._encoder_params_module_list = []
        self._non_encoder_params_module_list = []
        self._frozen_params_module_list = []

        self.ReturnType = namedtuple('ReturnType', ['depth_pred', 'bin_edges', 'detections'])

        self.logger.info("Building model")
        self.dense_feature_extractor = DenseFeatureExtractor(self.args)
        self.detector = Yolov7Wrapper(self.args)
        self.obj_lang_strategy = ObjectLanguageStrategy(self.args)
        self.language_model = LanguageEmbeddingWrapper(self.args)

        # Adding parts of the model to lists to make sure that the learning rate is adjusted/frozen appropriately.
        self._encoder_params_module_list.append(self.dense_feature_extractor.encoder)
        self._non_encoder_params_module_list.append(self.dense_feature_extractor.decoder)
        self._frozen_params_module_list.append(self.detector)
        self._frozen_params_module_list.append(self.obj_lang_strategy)
        self._frozen_params_module_list.append(self.language_model)
        
        max_seq_len = 1200 if self.args[self.args.model.name].get('do_final_upscale') else 500
        self.objcavit = ObjCAViT(
                            self.args,
                            n_query_channels=128, patch_size=16,
                            # image and object feature dims get projected to embedding_dim channels.
                            im_feature_dim=128,     # Image feature extractor output is 128d, so this should be 128d.
                            obj_feature_dim=self.language_model.get_num_object_features(),
                            embedding_dim=self.args[self.args.model.name].objcavit.embedding_dim,
                            dim_out=self.args.graphbins.n_bins,
                            norm='linear',
                            max_seq_len=max_seq_len,
                        )
        self._non_encoder_params_module_list.append(self.objcavit)

        self.conv_out = nn.Sequential(
                            nn.Conv2d(self.args[self.args.model.name].objcavit.embedding_dim, self.args.graphbins.n_bins, kernel_size=1, stride=1, padding=0),
                            nn.Softmax(dim=1)
                        )
        self._non_encoder_params_module_list.append(self.conv_out)
        

    def get_encoder_params(self):
        for m in self._encoder_params_module_list:
            yield from m.parameters()


    def get_non_encoder_params(self):
        for m in self._non_encoder_params_module_list:
            yield from m.parameters()

    
    def get_frozen_params(self):
        for m in self._frozen_params_module_list:
            yield from m.parameters()


    def forward(self, image):
        # Dense image features
        dense_features = self.dense_feature_extractor(image)

        self.detector.eval()
        self.detector.requires_grad_(False)
        self.language_model.eval()
        self.language_model.requires_grad_(False)
        # Object detection results. See fwd method of Yolov7Wrapper for documentation.
        with torch.no_grad():
            object_xywh_list, object_masks_list, object_confs_list, object_cls_list, object_names_list, annotated_images = self.detector(image)
            object_natural_language = self.obj_lang_strategy(
                                            object_xywh_list,
                                            object_masks_list,
                                            object_confs_list,
                                            object_cls_list,
                                            object_names_list,
                                        )
            names_features = self.language_model(
                                    class_num_list=object_cls_list,
                                    class_names=object_names_list,
                                    class_phrase_list=object_natural_language
                                )

        # Assemble object features
        object_features = [nf.float() for nf in names_features]
        bin_widths_normed, range_attention_maps = self.objcavit(dense_features, object_features, object_xywh_list)

        out = self.conv_out(range_attention_maps)	# Gives bin logits
        
        bin_widths = (self.args[self.args.basic.dataset].max_depth - self.args[self.args.basic.dataset].min_depth) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.args[self.args.basic.dataset].min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        depth_pred = torch.sum(out * centers, dim=1, keepdim=True)

        return self.ReturnType(depth_pred=depth_pred, bin_edges=bin_edges, detections=annotated_images)