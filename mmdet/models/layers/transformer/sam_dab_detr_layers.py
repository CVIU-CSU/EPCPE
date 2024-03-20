# Copyright (c) OpenMMLab. All rights reserved.
from typing import List,Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import ModuleList
from torch import Tensor

from .detr_layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DetrTransformerEncoder, DetrTransformerEncoderLayer)
from .utils import (MLP, ConditionalAttention, coordinate_to_encoding,
                    inverse_sigmoid)
from .two_way_transformer import TwoWayAttentionBlock,Attention
from .common import MLPBlock
from mmdet.utils import ConfigType, OptConfigType
from ....utils import get_root_logger



class DABDetrTwoWayTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Implements decoder layer in DAB-DETR transformer."""

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,),
                 conditional_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     attn_drop=0.,
                     proj_drop=0.,
                     cross_attn=True),
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     downsample_rate=2),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 mlp_cfg: OptConfigType = dict(
                    embed_dims=256,
                    mlp_dim=20,
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None,
                 is_last = False) -> None:

        super(DetrTransformerDecoderLayer,self).__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        self.conditional_attn_cfg = conditional_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg

        self.ffn_cfg = ffn_cfg
        self.mlp_cfg = mlp_cfg
        self.norm_cfg = norm_cfg
        self.is_last = is_last
        self._init_layers()


    def _init_layers(self):
        """Initialize self-attention, cross-attention, FFN, normalization and
        others."""
        self.self_attn = Attention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.norm1 = nn.LayerNorm(self.embed_dims)
        self.cross_attn_token_to_image = ConditionalAttention(**self.conditional_attn_cfg)
        self.norm2 = nn.LayerNorm(self.embed_dims)

        self.mlp = MLPBlock(**self.mlp_cfg,act=nn.ReLU)
        self.norm3 = nn.LayerNorm(self.embed_dims)

        self.keep_query_pos = self.cross_attn_token_to_image.keep_query_pos
        self.norm4 = nn.LayerNorm(self.embed_dims)
        self.cross_attn_image_to_token = Attention(**self.cross_attn_cfg)

        if self.is_last:
            self.final_attn_token_to_image = Attention(**self.cross_attn_cfg)
            self.norm_final_attn = nn.LayerNorm(self.embed_dims)

    def forward(
        self, query: Tensor, key: Tensor, query_pos: Tensor, key_pos: Tensor,
                ref_sine_embed: Tensor = None,
                self_attn_masks: Tensor = None,
                cross_attn_masks: Tensor = None,
                key_padding_mask: Tensor = None,
                is_first: bool = False,
                **kwargs
    ) -> Tensor:
        # Self attention block
        if is_first:
            query = self.self_attn(q=query, k=query, v=query)
        else:
            q = query + query_pos
            attn_out = self.self_attn(q=q, k=q, v=query)
            query = query + attn_out
        query = self.norm1(query)

        # Cross attention block, tokens attending to image embedding
        q = query + query_pos
        k = key + key_pos
        attn_out = self.cross_attn_token_to_image(
            query=query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            ref_sine_embed=ref_sine_embed,
            attn_mask=cross_attn_masks,
            key_padding_mask=key_padding_mask,
            is_first=is_first,
            **kwargs)
        query = query + attn_out
        query = self.norm2(query)

         # MLP block
        mlp_out = self.mlp(query)
        query = query + mlp_out
        query = self.norm3(query)

        q = query + query_pos
        k = key + key_pos
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=query)
        key = key + attn_out
        key = self.norm4(key)

        # Cross attention block, image embedding attending to tokens
        if self.is_last:
            q = query + query_pos
            k = key + key_pos
            attn_out=self.final_attn_token_to_image(q=q, k=k, v=key)
            query = query + attn_out
            query = self.norm_final_attn(query)
            

        return query, key


class DABDetrTwoWayTransformerDecoder(DetrTransformerDecoder):
    """Decoder of DAB-DETR.

    Args:
        query_dim (int): The last dimension of query pos,
            4 for anchor format, 2 for point format.
            Defaults to 4.
        query_scale_type (str): Type of transformation applied
            to content query. Defaults to `cond_elewise`.
        with_modulated_hw_attn (bool): Whether to inject h&w info
            during cross conditional attention. Defaults to True.
    """

    def __init__(self,
                 *args,
                 query_dim: int = 4,
                 query_scale_type: str = 'cond_elewise',
                 with_modulated_hw_attn: bool = True,
                 pretrained=None,
                 **kwargs):

        self.query_dim = query_dim
        self.query_scale_type = query_scale_type
        self.with_modulated_hw_attn = with_modulated_hw_attn
        self.pretrained=pretrained

        super().__init__(*args, **kwargs)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.pretrained is None:
            self.apply(_init_weights)
        elif isinstance(self.pretrained, str):
            #self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model decoder from: {self.pretrained}')
            ## Load SAM-B pretrained weights
            assert self.pretrained in {"SAM-B","SAM-L","SAM-H"}
            if self.pretrained=="SAM-B":
                state_dict = torch.load('pretrained/sam_vit_b_01ec64.pth')
            elif self.pretrained=="SAM-L":
                state_dict = torch.load('pretrained/sam_vit_l_0b3195.pth')
            elif self.pretrained=="SAM-H":
                state_dict = torch.load('pretrained/sam_vit_h_4b8939.pth')
            logger.info(f'load model from: {self.pretrained}')
            model_state_dict = {}
            for key,value in state_dict.items():
                if 'mask_decoder' in key:
                    #print(key)
                    new_key = key.replace('mask_decoder.transformer.','')
                    new_key = new_key.replace('mask_decoder.','') #去除前缀
                    if 'final' in new_key:
                        new_key = 'layers.1.'+new_key # 加上前缀
                    if 'cross_attn_token_to_image' in new_key:
                        new_key = new_key.replace('cross_attn_token_to_image','dont_load') #不导入这个参数，因为shape不一致
                    model_state_dict[new_key] = value
            
            msg = self.load_state_dict(model_state_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        else:
            raise TypeError('pretrained must be a str or None')

    def _init_layers(self):
        """Initialize decoder layers and other layers."""
        assert self.query_dim in [2, 4], \
            f'{"dab-detr only supports anchor prior or reference point prior"}'
        assert self.query_scale_type in [
            'cond_elewise', 'cond_scalar', 'fix_elewise'
        ]

        self.layers = ModuleList([
            DABDetrTwoWayTransformerDecoderLayer(**self.layer_cfg,is_last=(i==self.num_layers-1))
            for i in range(self.num_layers)
        ])

        embed_dims = self.layers[0].embed_dims
        self.embed_dims = embed_dims

        self.post_norm = build_norm_layer(self.post_norm_cfg, embed_dims)[1]
        if self.query_scale_type == 'cond_elewise':
            self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)
        elif self.query_scale_type == 'cond_scalar':
            self.query_scale = MLP(embed_dims, embed_dims, 1, 2)
        elif self.query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(self.num_layers, embed_dims)
        else:
            raise NotImplementedError('Unknown query_scale_type: {}'.format(
                self.query_scale_type))

        self.ref_point_head = MLP(self.query_dim // 2 * embed_dims, embed_dims,
                                  embed_dims, 2)

        if self.with_modulated_hw_attn and self.query_dim == 4:
            self.ref_anchor_head = MLP(embed_dims, embed_dims, 2, 2)

        self.keep_query_pos = self.layers[0].keep_query_pos
        if not self.keep_query_pos:
           for layer_id in range(self.num_layers - 1):
               self.layers[layer_id + 1].cross_attn_token_to_image.qpos_proj = None

    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Tensor,
                key_pos: Tensor,
                reg_branches: nn.Module,
                key_padding_mask: Tensor = None,
                **kwargs) -> List[Tensor]:
        """Forward function of decoder.

        Args:
            query (Tensor): The input query with shape (bs, num_queries, dim).
            key (Tensor): The input key with shape (bs, num_keys, dim).
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            reg_branches (nn.Module): The regression branch for dynamically
                updating references in each layer.
            key_padding_mask (Tensor): ByteTensor with shape (bs, num_keys).
                Defaults to `None`.

        Returns:
            List[Tensor]: forwarded results with shape (num_decoder_layers,
            bs, num_queries, dim) if `return_intermediate` is True, otherwise
            with shape (1, bs, num_queries, dim). references with shape
            (num_decoder_layers, bs, num_queries, 2/4).
        """
        output = query
        unsigmoid_references = query_pos

        reference_points = unsigmoid_references.sigmoid()
        intermediate_reference_points = [reference_points]

        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]
            ref_sine_embed = coordinate_to_encoding(
                coord_tensor=obj_center, num_feats=self.embed_dims // 2)
            query_pos = self.ref_point_head(
                ref_sine_embed)  # [bs, nq, 2c] -> [bs, nq, c]
            # For the first decoder layer, do not apply transformation
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]
            # apply transformation
            ref_sine_embed = ref_sine_embed[
                ..., :self.embed_dims] * pos_transformation
            # modulated height and weight attention
            if self.with_modulated_hw_attn:
                assert obj_center.size(-1) == 4
                ref_hw = self.ref_anchor_head(output).sigmoid()
                ref_sine_embed[..., self.embed_dims // 2:] *= \
                    (ref_hw[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                ref_sine_embed[..., : self.embed_dims // 2] *= \
                    (ref_hw[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            output,key = layer(
                output,
                key,
                query_pos=query_pos,
                ref_sine_embed=ref_sine_embed,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                is_first=(layer_id == 0),
                **kwargs)
            # iter update
            tmp_reg_preds = reg_branches(output)
            tmp_reg_preds[..., :self.query_dim] += inverse_sigmoid(
                reference_points)
            new_reference_points = tmp_reg_preds[
                ..., :self.query_dim].sigmoid()
            if layer_id != self.num_layers - 1:
                intermediate_reference_points.append(new_reference_points)
            reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.post_norm(output))

        output = self.post_norm(output)

        if self.return_intermediate:
            return [
                torch.stack(intermediate),
                torch.stack(intermediate_reference_points),
            ]
        else:
            return [
                output.unsqueeze(0),
                torch.stack(intermediate_reference_points)
            ]


class DABDetrTransformerEncoder(DetrTransformerEncoder):
    """Encoder of DAB-DETR."""

    def _init_layers(self):
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        embed_dims = self.layers[0].embed_dims
        self.embed_dims = embed_dims
        self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs):
        """Forward function of encoder.

        Args:
            query (Tensor): Input queries of encoder, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional embeddings of the queries, has
                shape (bs, num_feat_points, dim).
            key_padding_mask (Tensor): ByteTensor, the key padding mask
                of the queries, has shape (bs, num_feat_points).

        Returns:
            Tensor: With shape (num_queries, bs, dim).
        """

        for layer in self.layers:
            pos_scales = self.query_scale(query)
            query = layer(
                query,
                query_pos=query_pos * pos_scales,
                key_padding_mask=key_padding_mask,
                **kwargs)

        return query
