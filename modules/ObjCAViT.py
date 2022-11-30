# ObjCAViT.py
# Object Cross-Attention ViT.
# Accepts dense image features and list of object features.
# Applies a mixture of self-attention and cross-attention to these two inputs.
# Code implementation adapted from miniViT from the AdaBins repository.

import os, sys, logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
import math

from modules.layers import PatchTransformerEncoder, PixelWiseDotProduct


class GridRandomPositionalEmbeddings(nn.Module):
    """Implements one random, learnable vector per patch of image. For use with the object features,
                # the grid of positional embeddings is bilinearly upsampled back to full image resolution, and 
                # the required points are sampled from it.
    """
    def __init__(self, args, embedding_dim, patch_size, mode="centre"):
        """ Mode can be "centre", to just get the positional embeddings for the centre pixels, or
        "roi_align" to extract positional embeddings for a whole bounding box"""
        super().__init__()
        self.args = args
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.mode = mode

        assert self.mode in ["centre", "roi_align"], "Error: unrecognised GridRandomPositionalEmbeddings mode."

        # Work out the resultant size of the positional embedding based on the maximum image size.
        im_dims_train = self.args[self.args.basic.dataset].dimensions_train
        im_dims_test = self.args[self.args.basic.dataset].dimensions_test
        
        im_grid_size_train = [im_dims_train[0] / self.patch_size, im_dims_train[1] / self.patch_size]
        im_grid_size_test = [im_dims_test[0] / self.patch_size, im_dims_test[1] / self.patch_size]
        im_grid_size_train = [math.ceil(im_grid_size_train[0]), math.ceil(im_grid_size_train[1])]
        im_grid_size_test = [math.ceil(im_grid_size_test[0]), math.ceil(im_grid_size_test[1])]
        total_im_grid_train = im_grid_size_train[0] * im_grid_size_train[1]
        total_im_grid_test = im_grid_size_test[0] * im_grid_size_test[1]

        # This is the total number of random embeddings that will be used.
        self.sequence_length = max(total_im_grid_train, total_im_grid_test)
        self.positional_encodings = nn.Parameter(torch.rand(self.sequence_length, self.embedding_dim), requires_grad=True)


    def forward(self, coords, image_features, input_coord_space="img", factor=2.0):
        """
        Internal embeddings are same res as image_features.
        If input_coord_space == "img", will return one of these.
        If input_coord_space == "obj", will:
            1. interpolate self.positional_encodings up to full resolution, then will
            2. sample the resultant full-resolution image embeddings using the coords, and return those embeddings.

        Args:
            coords (torch.Tensor, Bx2): Batch of coordinates (x, y) to get embedding for. Assumed to be at full resolution.
            image_features (torch.Tensor, Bxembedding_dimxHxW): Image features from the dense encoder. 
                                                                Assumed to be at half resolution by default.
            input_coord_space (string, either "img" or "obj"): whether the input is in full resolution (object coords) or 
                                                               image feature resolution from the 
            factor (float, default 2.0): How much smaller the image_features are than full resolution.

        Returns:
            embeddings (torch.Tensor, Bxembedding_dim): Batch of embeddings, one for each point in the input batch.
        """

        # Select the number of positional encodings required based on the image size. Upsample + interpolate to
        # get pixelwise positional embeddings, then select those.
        # Full image resolution: [curr_im_height, curr_im_width] = HxW
        # image_features = BxCx(H/factor)x(W/factor) (from the dense image feature extractor - normally outputs half res.)
        # Object coords are from space HxW
        # Image feature coords (NOT image_features) are from space (H/factor/patch_size)x(W/factor/patch_size)
        curr_im_height = image_features.shape[2] * factor
        curr_im_width = image_features.shape[3] * factor
        curr_im_patch_feat_grid_height = math.ceil(image_features.shape[2] / self.patch_size)
        curr_im_patch_feat_grid_width = math.ceil(image_features.shape[3] / self.patch_size)

        curr_im_patch_feat_grid_len = curr_im_patch_feat_grid_height * curr_im_patch_feat_grid_width
        relevant_positional_encodings = self.positional_encodings[0:curr_im_patch_feat_grid_len, :]
        positional_encodings_grid = relevant_positional_encodings.view([curr_im_patch_feat_grid_height, curr_im_patch_feat_grid_width, self.positional_encodings.shape[1]]).permute(2, 0, 1).unsqueeze(0).contiguous()

        # positional_encodings_grid should now be a grid of positional embeddings, the same size as the image patch feature grid.
        # Normalise the input coordinates to fractions of the full resolution image (in range -1 to 1), then use those to sample the
        # smaller-resolution positional_encodings_grid

        match self.mode:
            case "centre":
                norm_coords = coords.clone()
                match input_coord_space:
                    case "img":
                        # Normalise coords relative to the positional encoding grid dimensions
                        norm_coords[:, 0] = ((norm_coords[:, 0] / curr_im_patch_feat_grid_height) * 2) - 1
                        norm_coords[:, 1] = ((norm_coords[:, 1] / curr_im_patch_feat_grid_width) * 2) - 1
                        norm_coords = norm_coords.unsqueeze(1)
                        positional_encodings_grid = positional_encodings_grid.expand(norm_coords.shape[0], -1, -1, -1)
                        samples = F.grid_sample(input=positional_encodings_grid, grid=norm_coords)
                        samples = samples.squeeze(2).permute(0, 2, 1).contiguous()

                    case "obj":
                        # Normalise coords relative to the full-resolution of the input image
                        norm_coords[:, 0] = ((norm_coords[:, 0] / curr_im_height) * 2) - 1
                        norm_coords[:, 1] = ((norm_coords[:, 1] / curr_im_width) * 2) - 1
                        norm_coords = norm_coords.view(1, 1, norm_coords.shape[0], 2)
                        
                        # Now the coords are normalised appropriately, and can be used to sample the positional encoding grid.
                        samples = F.grid_sample(input=positional_encodings_grid, grid=norm_coords)
                        samples = samples.squeeze(2).squeeze(0).permute(1, 0).contiguous()
            case "roi_align":
                # First convert input xywh (called "coords") to x1y1x2y2, where 0 <= x1 < x2 and 0 <= y1 < y2.
                match input_coord_space:
                    case "img":
                        xyxys = coords.clone()
                        half_widths = coords[:, :, 2] / 2
                        half_heights = coords[:, :, 3] / 2
                        xyxys[:, :, 0] = coords[:, :, 0] - half_widths
                        xyxys[:, :, 1] = coords[:, :, 1] - half_heights
                        xyxys[:, :, 2] = coords[:, :, 0] + half_widths
                        xyxys[:, :, 3] = coords[:, :, 1] + half_heights
                        
                        # Aggressively clip to zero to stop roi_align breaking
                        xyxys = torch.clamp(xyxys, min=0.0)
                        xyxys = [b for b in xyxys]
                        samples = []
                        for xyxy in xyxys:
                            sample = torchvision.ops.ps_roi_align(positional_encodings_grid.expand(len(xyxys), -1, -1, -1), [xyxy], output_size=[1, 1], spatial_scale=1 / self.patch_size)
                            samples.append(sample.squeeze(-1).squeeze(-1))

                        samples = torch.stack(samples, dim=0)

                    case "obj":
                        xyxys = coords.clone()
                        half_widths = coords[:, 2] / 2
                        half_heights = coords[:, 3] / 2
                        xyxys[:, 0] = coords[:, 0] - half_widths
                        xyxys[:, 1] = coords[:, 1] - half_heights
                        xyxys[:, 2] = coords[:, 0] + half_widths
                        xyxys[:, 3] = coords[:, 1] + half_heights
                        
                        # Aggressively clip to zero to stop roi_align breaking
                        xyxys = torch.clamp(xyxys, min=0.0)
                        samples = torchvision.ops.ps_roi_align(positional_encodings_grid, [xyxys], output_size=[1, 1], spatial_scale=1 / (self.patch_size * factor))
                        samples = samples.squeeze(-1).squeeze(-1)

        return samples


class SelfAttnCrossAttn(pl.LightningModule):
    def __init__(self, args, embedding_dim=128, num_heads=4, dim_feedforward=1024):
        super().__init__()
        self.args = args
        # Image transformer for image patch feature self-attention
        self.image_encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.image_transformer_encoder = nn.TransformerEncoder(self.image_encoder_layers, num_layers=4)  # takes shape S,N,E

        # Object transformer for object feature self-attention
        if self.args.graphbins.objcavit.get("no_obj_sa") != True:
            self.obj_encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=dim_feedforward, batch_first=True)
            self.obj_transformer_encoder = nn.TransformerEncoder(self.obj_encoder_layers, num_layers=4)  # takes shape S,N,E

        self.cross_attn_obj_im = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
        self.cross_attn_im_obj = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)


    def forward(self, image_patch_embeddings, object_features):
        # Image feature self-attention
        attended_image_features = self.image_transformer_encoder(image_patch_embeddings)    # Outputs BxSxC.

        # Attempt to apply cross-attention from self-attended object features to self-attended image patch features.
        # K is attended_object_features.
        # Q and V from attended_image_features

        # Applying self-attention to both object and image features
        # Object feature self-attention (if present)
        # Pad all things in object_features and generate masks, to allow batched inference
        # src_key_padding_mask for Transformer: True values get padded out (attn value set to -inf), False means NOT ignored/padded.
        # I.e. True means "this value is a padding value"
        obj_feature_masks = [torch.zeros(obj.shape[0], device=image_patch_embeddings.device).bool() for obj in object_features]
        obj_feature_masks = nn.utils.rnn.pad_sequence(obj_feature_masks, batch_first=True, padding_value=True)
        # obj_feature_masks |= torch.logical_not(is_real_object_mask.unsqueeze(1))   # Set to True all invalid objs.
        object_features = nn.utils.rnn.pad_sequence(object_features, batch_first=True, padding_value=0.0001)

        if self.args.graphbins.objcavit.get("no_obj_sa") == True:
            attended_object_features = object_features
        else:
            attended_object_features = self.obj_transformer_encoder(object_features, src_key_padding_mask=obj_feature_masks)

        if attended_object_features is not None:
            # Pad attended_object_features and the masks to the length of the attended_image_features
            amt_to_pad = attended_image_features.shape[1] - attended_object_features.shape[1]
            key_padding_mask = F.pad(obj_feature_masks, pad=(0, amt_to_pad), value=True)
            attended_object_features = F.pad(attended_object_features, pad=(0, 0, amt_to_pad, 0), value=0.0001)
            final_image_features, _ = self.cross_attn_obj_im(
                                        query=attended_image_features,
                                        key=attended_object_features,
                                        value=attended_image_features,
                                        key_padding_mask=key_padding_mask,
                                        need_weights=False,
                                    )
            final_object_features, _ = self.cross_attn_im_obj(
                                        query=attended_object_features,
                                        key=attended_image_features,
                                        value=attended_object_features,
                                        need_weights=False,
                                    )
        else:
            # In lieu of detected objects, just do nothing (i.e. whole of attended_object_features gets masked).
            final_image_features = attended_image_features
            final_object_features = attended_object_features
        
        return final_image_features, final_object_features


class ObjCAViT(pl.LightningModule):
    """Implements the Object Cross-Attention Vision Transformer block.
    This block takes in dense image features and detected object features, and applies attention:
    Self-attention is applied to the image features to relate each image feature to each other image feature,
    and self-attention is applied to the object features to relate each object to each other object.

    Cross-attention is then applied between the two to relate the visual features to the semantics of the objects
    detected in the image, as well as the relationships between those objects.

    Operation:
        1. Image patches are extracted and embedded to embedding_dim dimensions.
        2. Object features, if present, are embedded to embedding_dim dimensions.
        3. Positional embeddings are computed for both from coordinates, and added on to the features.
        4. Self-attention is applied separately to both the object features and the image features
        5. Cross-attention is applied from the object features to the image features
        6. The results cross-attended image features are used to compute the adaptive bins
    """
    def __init__(self, args, im_feature_dim=128, obj_feature_dim=512, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear', max_seq_len=50):
        super().__init__()
        self.args = args
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_size = patch_size
        self.half_patch_size = self.patch_size // 2     # Convenience, to avoid repeatedly computing the division
        self.obj_feature_dim = obj_feature_dim

        # Build positional embedding strategy. Each should intake Bx2 coordinates, and output Bxembedding_dim positional features.
        match self.args[self.args.model.name].objcavit.positional_embedding_strategy:
            case "grid_random":
                # Implements one random, learnable vector per patch of image. For use with the object features,
                # the grid of positional embeddings is bilinearly upsampled back to full image resolution, and 
                # the required points are sampled from it.
                self.positional_encoder = GridRandomPositionalEmbeddings(args, embedding_dim=embedding_dim, patch_size=patch_size, mode="centre")
            case "grid_random_roi_align":
                # As grid_random, but gets positional features for the whole of the bbox supplied (or whole image patch, in the case of the image
                # features). Aim is to capture positional information for the entirety of the bbox, therefore including size information.
                self.positional_encoder = GridRandomPositionalEmbeddings(args, embedding_dim=embedding_dim, patch_size=patch_size, mode="roi_align")
            case "learned":
                # Uses an MLP to embed the 2d coordinates into a high-dimensional space
                # Architecture is similar to SuperGlue (Sarlin et al. 2020)
                self.positional_encoder = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.LeakyReLU(),
                    nn.Linear(32, 64, bias=True),
                    nn.LeakyReLU(),
                    nn.Linear(64, 128, bias=True),
                    nn.LeakyReLU(),
                    nn.Linear(128, 256, bias=True),
                    nn.LeakyReLU(),
                    nn.Linear(256, embedding_dim, bias=True)
                )
            case "learned_bbox_wh":
                # Exactly the same as "learned", but with 4 channels on the input.
                # The extra 2 channels are for the bbox width and height.
                # When getting positional embeddings for visual patches, width and height are for the patch.
                self.positional_encoder = nn.Sequential(
                    nn.Linear(4, 32, bias=True),
                    nn.LeakyReLU(),
                    nn.Linear(32, 64, bias=True),
                    nn.LeakyReLU(),
                    nn.Linear(64, 128, bias=True),
                    nn.LeakyReLU(),
                    nn.Linear(128, 256, bias=True),
                    nn.LeakyReLU(),
                    nn.Linear(256, embedding_dim, bias=True)
                )
            case _:
                sys.exit("Error: ObjCAViT positional embedding strategy not recognised.")

        # Image and object feature networks: Conv2d for image patches (gets flattened), FFN for obj. features
        self.image_embedding_convPxP = nn.Conv2d(im_feature_dim, embedding_dim,
                                           kernel_size=self.patch_size, stride=patch_size, padding=0)
        self.obj_embedding_layer = nn.Linear(self.obj_feature_dim, embedding_dim)

        # Defining self-attention and cross-attention layers.
        self.saca_1 = SelfAttnCrossAttn(self.args, embedding_dim, num_heads, dim_feedforward=1024)
        if self.args.graphbins.objcavit.get("use_2_saca") == True:
            self.saca_2 = SelfAttnCrossAttn(self.args, embedding_dim, num_heads, dim_feedforward=1024)

        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(im_feature_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))


    def forward(self, image_features, object_features, object_xywh_list):
        # 1: Moving image and object features to a common space, and adding positional embeddings
        # Get object positional embeddings and add to object features
        obj_pos_embeddings = []
        is_real_object_mask = torch.tensor([feat is not None for feat in object_xywh_list], device=image_features.device)
        for i, xywhs in enumerate(object_xywh_list):
            if xywhs is None:
                xywhs = torch.zeros((1, 4), device=image_features.device) - 1   # Assign impossible coords and sizes
                # Don't need to assign object_features because they're embeddings of "<UNK>"
                # If no detections, there's 1 UNK per image, so object_features[i] has shape 1xobj_feature_dim
            
            # Get object positional embeddings (according to strategy)
            match self.args[self.args.model.name].objcavit.positional_embedding_strategy:
                case "grid_random":
                    obj_pos_embeddings.append(self.positional_encoder(xywhs[:, 0:2], image_features, "obj"))
                case "grid_random_roi_align":
                    obj_pos_embeddings.append(self.positional_encoder(xywhs[:, 0:4], image_features, "obj"))
                case "learned":
                    obj_pos_embeddings.append(self.positional_encoder(xywhs[:, 0:2]))
                case "learned_bbox_wh":
                    obj_pos_embeddings.append(self.positional_encoder(xywhs[:, 0:4]))
                case _: # Default case: try to embed just the coords.
                    obj_pos_embeddings.append(self.positional_encoder(xywhs[:, 0:2]))

            object_features[i] = self.obj_embedding_layer(object_features[i]) + obj_pos_embeddings[i]

        # Do the same for the image features, but first split into patches and get patch features
        image_patch_embeddings = self.image_embedding_convPxP(image_features.clone())

        # Making coordinate tensor and expanding to image feature dimensions
        patch_coords_w = torch.tensor(range(image_patch_embeddings.shape[3]), device=image_features.device).view(1, -1)
        patch_coords_w = patch_coords_w.expand(image_patch_embeddings.shape[2], -1)
        patch_coords_h = torch.tensor(range(image_patch_embeddings.shape[2]), device=image_features.device).view(-1, 1)
        patch_coords_h = patch_coords_h.expand(-1, image_patch_embeddings.shape[3])
        patch_coords = torch.stack([patch_coords_w, patch_coords_h], dim=0)
        patch_coords = (patch_coords * self.patch_size) + self.half_patch_size  # Centre coords of each patch, inc. padding.
        patch_coords = patch_coords.flatten(1)  # 2xN, where N == image_patch_embeddings.flatten(2).shape[2]
        patch_coords = patch_coords.expand(image_features.shape[0], -1, -1).permute(0, 2, 1).float()

        # Make the patch sizes tensor, for use when w/h are embedded somehow.
        patch_sizes = torch.ones_like(patch_coords, device=image_features.device) * self.patch_size
        patch_coords = torch.cat([patch_coords, patch_sizes], dim=2)

        # Adding image patch positional embeddings to image patch features
        match self.args[self.args.model.name].objcavit.positional_embedding_strategy:
            case "grid_random":
                image_pos_embeddings = self.positional_encoder(patch_coords[..., 0:2], image_features, "img")
            case "grid_random_roi_align":
                image_pos_embeddings = self.positional_encoder(patch_coords[..., 0:4], image_features, "img")
            case "learned":
                image_pos_embeddings = self.positional_encoder(patch_coords[..., 0:2])
            case "learned_bbox_wh":
                image_pos_embeddings = self.positional_encoder(patch_coords[..., 0:4])
            case _: # Default case: try to embed just the coords.
                image_pos_embeddings = self.positional_encoder(patch_coords[..., 0:2])

        image_pos_embeddings = image_pos_embeddings.permute(0, 2, 1).contiguous()
        image_patch_embeddings = image_patch_embeddings.flatten(2) + image_pos_embeddings
        image_patch_embeddings = image_patch_embeddings.permute(0, 2, 1)    # transformer w/ batch_first requires BxSxC.

        image_patch_embeddings, object_features = self.saca_1(image_patch_embeddings, object_features)
        if self.args.graphbins.objcavit.get("use_2_saca") == True:
            image_patch_embeddings, object_features = self.saca_2(image_patch_embeddings, object_features)
            
        final_image_features = image_patch_embeddings
        final_object_features = object_features

        regression_head, queries = final_image_features[:, 0, :], final_image_features[:, 1:self.n_query_channels + 1, :]
        image_features = self.conv3x3(image_features)
        range_attention_maps = self.dot_product_layer(image_features, queries)  # .shape = n, n_query_channels, h, w

        # Bin centre regression
        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)

        y = y / y.sum(dim=1, keepdim=True)

        return y, range_attention_maps

