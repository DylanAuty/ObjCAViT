# LanguageEmbeddingWrapper.py
# Wrapper for different language embeddings, to provide a clean API to use:
# Expects object detection results (class, names, and natural_language thing to use)
# Outputs list of features of length B, each of size Nxobject embedding dim.

import os, sys, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from modules.CLIPWrapper import CLIPWrapper

class LanguageEmbeddingWrapper(pl.LightningModule):
    """
    Wrapper for different language embeddings, to provide a clean API to use:
    Expects object detection results (class, names, and natural_language thing to use)
    Outputs list of features of length B, each of size Nxobject embedding dim.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.strat = self.args[self.args.model.name].objcavit.language_embedding_strategy

        match self.strat:
            case "control_obj_zeros_512":
                pass    # for readability
            case "clip":
                self.language_model = CLIPWrapper(self.args)
            case _:
                sys.exit(f"Error: Language model {self.strat} not recognised")

    
    def get_num_classes(self):
        """ Returns the (probable) number of classes that the object detector was trained on. """
        if "lvis" in self.args[self.args.model.name].yolov7_chkpt:
            return 1204 # Assuming LVIS v1.0 (1203), plus 1 because of how yolov7 works
        else:
            return 80


    def get_num_object_features(self):
        """Returns number of object features that this module will output, which will vary
        based on the language model used.
        """
        match self.strat:
            case "control_obj_zeros_512":
                return 512
            case "clip":
                return 512  # by default, CLIP uses ViT-B/32 which has 512d embeddings.
            case _:
                sys.exit(f"Error: Language model {self.strat} not recognised")

    
    def forward(self, class_num_list, class_names, class_phrase_list):
        match self.strat:
            case "control_obj_zeros_512":
                names_features = []
                for cls in class_num_list:
                    if cls is None:
                        cls = torch.tensor([0], device=self.device, dtype=torch.int64)
                    names_features.append(torch.zeros([len(cls), 512], device=self.device))
            case "clip":
                names_features = self.language_model(class_phrase_list) # list of length B, each element is Nx512 (N=number of detections.)
            case _:
                sys.exit(f"Error: Language model {self.strat} not recognised")
        
        return names_features