# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

from mmengine.model import uniform_init
from torch import Tensor, nn

from mmdet.registry import MODELS
from ..layers import SinePositionalEncoding
from ..layers.transformer import (DABDetrTransformerDecoder,
                                  DABDetrTransformerEncoder, inverse_sigmoid)
from .detr import DETR
from mmdet.structures import OptSampleList, SampleList
from mmengine.visualization import Visualizer
import numpy as np
import mmcv
import torch
#img_path= '/root/commonfile/fxf/nocs/CAMERA/val/00000/0000_color.png'

@MODELS.register_module()
class DABDETRPhoCal(DETR):
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
                 **kwargs) -> None:
        self.with_random_refpoints = with_random_refpoints
        assert isinstance(num_patterns, int), \
            f'num_patterns should be int but {num_patterns}.'
        self.num_patterns = num_patterns

        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DABDetrTransformerEncoder(**self.encoder)
        self.decoder = DABDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_dim = self.decoder.query_dim
        self.query_embedding = nn.Embedding(self.num_queries, self.query_dim)
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DABDETRPhoCal, self).init_weights()
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
        query_pos = self.query_embedding.weight
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

        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, memory_pos: Tensor) -> Dict:
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

        hidden_states, references = self.decoder(
            query=query,
            key=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask,
            reg_branches=self.bbox_head.fc_reg  # iterative refinement for anchor boxes
        )
        head_inputs_dict = dict(
            hidden_states=hidden_states, references=references)
        return head_inputs_dict

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs)
        img_feats = x
        if self.with_neck:
            x = self.neck(x)

        # print(img_feats[0].shape)
        # img = mmcv.imread(img_path, channel_order='rgb')
        # backbone_visualizer=Visualizer(vis_backends=[dict(type='LocalVisBackend')],
        #                             save_dir='tmp')
        # # print(image[0].permute(1,2,0),image[0].permute(1,2,0).shape)
        # # print(x[0],x[0].shape)
        # backbone_img=backbone_visualizer.draw_featmap(img_feats[0][0],img, channel_reduction='select_max')
        # #visualizer.show(visual_img)
        # backbone_visualizer.add_image('backbone', backbone_img,22)

        return x

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        #image=batch_inputs
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        # print(img_feats[0].shape)
        # img = mmcv.imread(img_path, channel_order='rgb')
        # backbone_visualizer=Visualizer(vis_backends=[dict(type='LocalVisBackend')],
        #                             save_dir='tmp')
        # # print(image[0].permute(1,2,0),image[0].permute(1,2,0).shape)
        # # print(x[0],x[0].shape)
        # backbone_img=backbone_visualizer.draw_featmap(img_feats[0][0],img, channel_reduction='squeeze_mean')
        # #visualizer.show(visual_img)
        # backbone_visualizer.add_image('dinov2 backbone', backbone_img,1)

        # visualizer=Visualizer(image=np.array(image[0].cpu().permute(1,2,0)),
        #         vis_backends=[dict(type='LocalVisBackend')],
        #                             save_dir='tmp')

        # # print(image[0].permute(1,2,0),image[0].permute(1,2,0).shape)
        # # print(x[0],x[0].shape)
        # print(head_inputs_dict)
        # print(head_inputs_dict['hidden_states'].shape)
        # #dab_img=visualizer.draw_featmap(head_inputs_dict['hidden_states'][0],img, channel_reduction='squeeze_mean')
        # #visualizer.show(visual_img)
        # visualizer.add_image('dab-detr', visualizer.get_image(),4)
        return batch_data_samples

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
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        # print(encoder_outputs_dict['memory'].shape)
        # print(encoder_outputs_dict['memory'].permute(0,2,1).reshape(-1,35,46).shape)
        # img = mmcv.imread(img_path, channel_order='rgb')
        # encoder_visualizer=Visualizer(vis_backends=[dict(type='LocalVisBackend')],
        #                             save_dir='tmp')
        # # print(image[0].permute(1,2,0),image[0].permute(1,2,0).shape)
        # print(img.shape)
        # # print(x[0],x[0].shape)
        # encoder_img=encoder_visualizer.draw_featmap(encoder_outputs_dict['memory'].permute(0,2,1).reshape(-1,35,46),img, channel_reduction='squeeze_mean')
        # #visualizer.show(visual_img)
        # encoder_visualizer.add_image('dinov2 encoder featmap', encoder_img,1)

        return head_inputs_dict
