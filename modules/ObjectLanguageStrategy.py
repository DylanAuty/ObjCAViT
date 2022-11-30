# ObjectLanguageStrategy.py
# Implements the various different object language strategies.
# Inputs object detections (labels, bbox centres + dimensions, masks)
# Outputs one natural language phrase per object, to be fed to the language model.

import math
import os, sys, logging
import torch
import torch.nn as nn
import numpy as np

from nltk.corpus import wordnet as wn
import re


class ObjectLanguageStrategy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.strat = self.args[self.args.model.name].objcavit.obj_language_strategy
        self.logger = logging.getLogger(f"{__name__}({self.strat})")

        self.seven_pt_nl_size_scale = [
            "much smaller than",
            "smaller than",
            "a bit smaller than",
            "about the same size as",
            "a bit bigger than",
            "bigger than",
            "much bigger than",
        ]

        # This is here for forward compatibility - may use different scales in future.
        self.rel_size_scale = self.seven_pt_nl_size_scale

        # Sanity checking:
        match self.strat:
            case "synset_def_wn":
                if "lvis" not in self.args[self.args.model.name].yolov7_chkpt:
                    self.logger.warning("Possible error: trying to use synset definition strategy with a non-LVIS-trained checkpoint.")

    def synset_to_name(self, synset):
        """ Converts a synset in the form "obj_name.n.01" to just the name "obj name". """
        synset = synset.split(".", 1)[0]
        synset = re.sub('[^a-zA-Z0-9 \.]', ' ', synset)
        return synset


    def get_single_relative_size_clause(self, object_xywh_list, object_masks_list, object_confs_list, object_cls_list, object_names_list):
        """ For each object in each image's detection list, generates a new clause that relates the apparent
        sizes of that object and one other object (the next in the list).
        If there are not objects or only one object detected in the image, returns an empty string.

        """
        relative_size_clauses = []
        if object_xywh_list is None:
            relative_size_clauses.append([[""]])
        else:
            for i, obj_list in enumerate(object_xywh_list):
                obj_rel_size_clauses = []
                if obj_list is None:
                    obj_rel_size_clauses.append("")
                else:
                    for j, obj_xywh in enumerate(obj_list):
                        clause = ""
                        if len(obj_list) <= 1:
                            obj_rel_size_clauses.append(clause)
                        else:
                            obj_bbox_area = obj_xywh[2] * obj_xywh[3]
                            next_obj_idx = (j + 1) % len(obj_list)
                            next_obj_xywh = obj_list[next_obj_idx]
                            next_obj_bbox_area = next_obj_xywh[2] * next_obj_xywh[3]
                            
                            # Assign to a clause based on relative scale (compared relative to the other object)
                            # Idea is that everything between 1/e and e times as big as the other object will be
                            # representable with the scale given, and differences greater than that will be 
                            rel_fac = math.log(obj_bbox_area / next_obj_bbox_area) + 1 # 0-2 is now the range of the valid bins
                            rel_fac /= 2    # now in range 0-1, so we map 0-1 to the middle bins of the array
                            rel_fac *= (len(self.rel_size_scale) - 3)     # 0-(middle bins length - 1)
                            rel_fac = np.round(rel_fac) + 1 # 1-len
                            rel_fac = int(np.clip(rel_fac, 0, len(self.rel_size_scale) - 1))

                            clause = self.rel_size_scale[rel_fac]
                            obj_name = self.synset_to_name(object_names_list[i][j])
                            other_obj_name = self.synset_to_name(object_names_list[i][next_obj_idx])
                            other_string = ""
                            if other_obj_name == obj_name:
                                other_string = "other "

                            clause = f"This {obj_name} appears {clause} the {other_string}{other_obj_name}"
                        obj_rel_size_clauses.append(clause)
                relative_size_clauses.append(obj_rel_size_clauses)
        return relative_size_clauses


    def get_synset_definition(self, term):
        """ Returns a definition for a single Wordnet synset.
        If an LVIS YOLOv7 checkpoint is being used, will look up synset definitions.
        If it fails to find the synset definition (e.g. class stop_sign.n.01 isn't actually a wordnet synset, but
        is still present in LVIS) then will either use a manually selected other definition, return the lemma, or
        return "<UNK>" as the definition.
        """
        definition = None
        if term is None:
            definition = "<UNK>"
        else:
            try:
                # There is one key in LVIS that isn't actually a wordnet synset: stop_sign.n.01.
                # This was checked by attempting to fetch definitions for all LVIS keys.
                # The unused label number 0 (with label "UNUSED LABEL") is also not found, but
                # as it doesn't occur in the dataset, it shouldn't be an issue.
                definition = wn.synset(term).definition()
            except:
                if term == "stop_sign.n.01":
                    # This definition taken from the first line of the English Wikipedia article about stop signs (https://en.wikipedia.org/wiki/Stop_sign, accessed 01/11/22)
                    definition = "A stop sign is a traffic sign designed to notify drivers that they must come to a complete stop and make sure the intersection is safely clear of vehicles and pedestrians before continuing past the sign."
                else:
                    # Only exceptions should still be formatted like synsets, so we extract the "lemma" and use that as the definition.
                    definition = self.synset_to_name(term)
        
        # Error catching.
        if definition is None:
            definition = "<UNK>"

        return definition


    def forward(self, object_xywh_list, object_masks_list, object_confs_list, object_cls_list, object_names_list):
        """Args:
            object_xywh_list (list of B torch.Tensors of size (Nix4)): item[:, 0:2] is xy coord (in pixels) of bbox centre.
                                                                       item[:, 2:4] is width and height of bbox (in px)
            object_masks_list (list of B torch.Tensors of size (NxCxHxW)): Instance masks.
            object_confs_list (list of B torch.Tensors of size (N): confidence of each detection
            object_cls_list  (list of B torch.Tensors of size (N): Class of detection
            object_names_list  (list of B lists of N strings): Labels of detected class.
                                                               If using LVIS-trained object detector, these will be 
                                                               wordnet synsets.
        """
        match self.strat:
            case "none":
                return [[name for name in nl] if nl is not None else ["<UNK>"] for nl in object_names_list]
            case "synset_def_wn":
                # If no detection present, returns "<UNK>".
                names_definitions = []  # This will be a list of length B. Each element will itself be a list of definitions.
                for i, nl in enumerate(object_names_list):
                    if nl is None:
                        names_definitions.append(["<UNK>"])
                    else:
                        nl_list = []
                        for j, synset in enumerate(nl):
                            definition = self.get_synset_definition(synset)
                            nl_list.append(definition)

                        names_definitions.append(nl_list)

            case "name_synset_def_wn_rel_sz":
                # If no detection present, returns "<UNK>".
                names_definitions = []  # This will be a list of length B. Each element will itself be a list of definitions.
                for i, nl in enumerate(object_names_list):
                    if nl is None:
                        names_definitions.append(["<UNK>"])
                    else:
                        nl_list = []
                        for j, synset in enumerate(nl):
                            definition = self.get_synset_definition(synset)
                            name = self.synset_to_name(synset)
                            nl_list.append(f"This is {'an' if name[0] in 'aeiou' else 'a'} {name}, defined as {definition}")

                        names_definitions.append(nl_list)
                clauses_to_add = self.get_single_relative_size_clause(object_xywh_list, object_masks_list, object_confs_list, object_cls_list, object_names_list)
                
                for i, def_list in enumerate(names_definitions):
                    for j, defn in enumerate(def_list):
                        names_definitions[i][j] = f"{defn}. {clauses_to_add[i][j]}."

            case _:
                sys.exit(f"ERROR: object language strategy not recognised: {self.strat}")
            
        return names_definitions
