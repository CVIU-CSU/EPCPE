# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

from mmengine.model import uniform_init
from torch import Tensor, nn
import torch
import torch.nn.functional as F

from mmdet.registry import MODELS
from ..layers import SinePositionalEncoding
from ..layers.transformer import (DABDetrTransformerDecoder,
                                  DABDetrTransformerEncoder, inverse_sigmoid)
from ..layers.transformer import SAMTwoWayTransformerDecoder
from .detr import DETR
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType


@MODELS.register_module()
class SAMDABADAPTER(DETR):
    r"""Implementation of `DAB-DETR:
    Dynamic Anchor Boxes are Better Queries for DETR.

    <https://arxiv.org/abs/2201.12329>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DAB-DETR>`_.

    Args:
        with_random_refpoints (bool): Whether to randomly initialize query
            embeddings and not update them during training.
            Defaults to False.
        num_patterns (int): Inspired by Anchor-DETR. Defaults to 0.
    """

    def __init__(self,
                 *args,
                 with_random_refpoints: bool = False,
                 num_patterns: int = 0,
                 sam_decoder: OptConfigType = None,
                 hs_sum_weight:float = None,
                 **kwargs) -> None:
        self.with_random_refpoints = with_random_refpoints
        assert isinstance(num_patterns, int), \
            f'num_patterns should be int but {num_patterns}.'
        self.num_patterns = num_patterns
        self.sam_decoder = sam_decoder

        super().__init__(*args, **kwargs)
        if hs_sum_weight==None:
            self.learned_gate=True
            self.hs_sum_weight= nn.Parameter(torch.zeros(1)) #from https://github.com/11yxk/SAM-LST/blob/8272e46da292437a3b3ae9268a11e2743048290f/segment_anything/modeling_CNN/sam.py#L25
            self.sigmoid = nn.Sigmoid()
        else:
            self.learned_gate=False
            self.hs_sum_weight=hs_sum_weight

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.sam_decoder = SAMTwoWayTransformerDecoder(**self.sam_decoder)
        self.dab_decoder = DABDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.dab_decoder.embed_dims
        self.query_dim = self.dab_decoder.query_dim
        self.query_embedding = nn.Embedding(self.num_queries, self.query_dim)
        self.query_embedding_sam = nn.Embedding(self.num_queries,self.embed_dims)
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DETR, self).init_weights()
        for p in self.dab_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.with_random_refpoints:
            uniform_init(self.query_embedding)
            self.query_embedding.weight.data[:, :2] = \
                inverse_sigmoid(self.query_embedding.weight.data[:, :2])
            self.query_embedding.weight.data[:, :2].requires_grad = False

    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword args dictionary of
                `self.forward_decoder()`, which includes 'query', 'query_pos',
                'memory' and 'reg_branches'.
            - head_inputs_dict (dict): The keyword args dictionary of the
                bbox_head functions, which is usually empty, or includes
                `enc_outputs_class` and `enc_outputs_class` when the detector
                support 'two stage' or 'query selection' strategies.
        """
        batch_size = memory.size(0)
        query_pos = self.query_embedding.weight #4
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.num_patterns == 0:
            query = query_pos.new_zeros(batch_size, self.num_queries,
                                        self.embed_dims)
        else:
            query = self.patterns.weight[:, None, None, :]\
                .repeat(1, self.num_queries, batch_size, 1)\
                .view(-1, batch_size, self.embed_dims)\
                .permute(1, 0, 2)
            query_pos = query_pos.repeat(1, self.num_patterns, 1)
        query_pos_sam = self.query_embedding_sam.weight #256
        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory ,query_pos_sam=query_pos_sam)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, query: Tensor, query_pos: Tensor, query_pos_sam:Tensor, 
                        memory: Tensor, memory_mask: Tensor, memory_pos: Tensor) -> Dict:
        """Forward with Transformer decoder.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            memory_pos (Tensor): The positional embeddings of memory, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` and `references` of the decoder output.
        """
        
        hidden_states_sam = self.sam_decoder(
            query=query,
            key=memory,
            query_pos=query_pos_sam,
            key_pos=memory_pos)  # for cross_attn

        hidden_states, references = self.dab_decoder(
            query=query,
            key=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask,
            reg_branches=self.bbox_head.fc_reg  # iterative refinement for anchor boxes
        )

        hidden_states_sam=(self.hs_sum_weight)*hidden_states_sam+(1-self.hs_sum_weight)*hidden_states
        
        head_inputs_dict = dict(
            hidden_states=hidden_states, references=references,
            hidden_states_sam=hidden_states_sam)
        return head_inputs_dict

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:

        .. code:: text

                 img_feats & batch_data_samples
                               |
                               V
                      +-----------------+
                      | pre_transformer |
                      +-----------------+
                          |          |
                          |          V
                          |    +-----------------+
                          |    | forward_encoder |
                          |    +-----------------+
                          |             |
                          |             V
                          |     +---------------+
                          |     |  pre_decoder  |
                          |     +---------------+
                          |         |       |
                          V         V       |
                      +-----------------+   |
                      | forward_decoder |   |
                      +-----------------+   |
                                |           |
                                V           V
                               head_inputs_dict

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        decoder_inputs_dict,feat = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict=dict(memory=feat) #去除encoder 直接把feat当作memory输入

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Prepare the inputs of the Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            img_feats (Tuple[Tensor]): Tuple of features output from the neck,
                has shape (bs, c, h, w).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such as
                `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask',
              and 'memory_pos'.
        """

        feat = img_feats[-1]  # NOTE img_feats contains only one feature.
        batch_size, feat_dim, _, _ = feat.shape
        # construct binary masks which for the transformer.
        #assert batch_data_samples is not None #for flops evaluate
        batch_input_shape = batch_data_samples[0].batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]

        input_img_h, input_img_w = batch_input_shape
        masks = feat.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values represent
        # ignored positions, while zero values mean valid positions.

        masks = F.interpolate(
            masks.unsqueeze(1), size=feat.shape[-2:]).to(torch.bool).squeeze(1)
        # [batch_size, embed_dim, h, w]
        pos_embed = self.positional_encoding(masks)

        # use `view` instead of `flatten` for dynamically exporting to ONNX
        # [bs, c, h, w] -> [bs, h*w, c]
        feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        # [bs, h, w] -> [bs, h*w]
        masks = masks.view(batch_size, -1)

        # prepare transformer_inputs_dict
        decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
        return decoder_inputs_dict,feat