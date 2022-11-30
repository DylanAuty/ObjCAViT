# CLIPWrapper.py
# Wrapper for CLIP, to be used for extracting word/sentence embeddings.

import os, sys, logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
import clip


class CLIPWrapper(pl.LightningModule):
    """
    A PyTorch Lightning LightningModule wrapper around CLIP, to be used to extract text embeddings.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model, self.preprocess = clip.load("ViT-B/32", device="cpu")

    
    def forward(self, phrase_list):
        tokenized = [clip.tokenize(pl).to(self.device) for pl in phrase_list]
        text_features = [self.model.encode_text(tkn) for tkn in tokenized]

        return text_features