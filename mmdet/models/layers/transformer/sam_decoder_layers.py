# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union,Tuple

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine import ConfigDict
from mmengine.model import BaseModule, ModuleList
from torch import Tensor,nn

from mmdet.utils import ConfigType, OptConfigType
from .two_way_transformer import Attention
from .common import MLPBlock
from ....utils import get_root_logger



class SAMTwoWayTransformerDecoder(BaseModule):
    """Decoder of DETR.

    Args:
        num_layers (int): Number of decoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        post_norm_cfg (:obj:`ConfigDict` or dict, optional): Config of the
            post normalization layer. Defaults to `LN`.
        return_intermediate (bool, optional): Whether to return outputs of
            intermediate layers. Defaults to `True`,
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 post_norm_cfg: OptConfigType = dict(type='LN'),
                 pretrained=None,
                 init_cfg: Union[dict, ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        self.post_norm_cfg = post_norm_cfg
        self.pretrained=pretrained
        self._init_layers()
    
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
                    model_state_dict[new_key] = value
            
            msg = self.load_state_dict(model_state_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        else:
            raise TypeError('pretrained must be a str or None')

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            SAMTwoWayTransformerDecoderLayer(**self.layer_cfg,is_last=(i==self.num_layers-1))
            for i in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, key: Tensor,
                query_pos: Tensor, key_pos: Tensor,
                **kwargs) -> Tensor:
        """Forward function of decoder
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor): The input key, has shape (bs, num_keys, dim).
            value (Tensor): The input value with the same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).

        Returns:
            Tensor: The forwarded results will have shape
            (num_decoder_layers, bs, num_queries, dim) if
            `return_intermediate` is `True` else (1, bs, num_queries, dim).
        """

        for layer_id,layer in enumerate(self.layers):
            query,key = layer(
                query=query,
                key=key,
                query_pos=query_pos,
                key_pos=key_pos,
                is_first =(layer_id == 0),
                **kwargs)

        output = query.unsqueeze(0) #在返回前要扩张一个维度 follow not return_intermediate的做法

        return output



class SAMTwoWayTransformerDecoderLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,),
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     downsample_rate=2),
                 mlp_cfg: OptConfigType = dict(
                    embed_dims=256,
                    mlp_dim=2048,
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None,
                 is_last = False) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg
        
        self.norm_cfg = norm_cfg
        self.mlp_cfg = mlp_cfg
        self.is_last = is_last
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = Attention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims

        self.norm1 = nn.LayerNorm(self.embed_dims)
        self.cross_attn_token_to_image = Attention(**self.cross_attn_cfg)
        self.norm2 = nn.LayerNorm(self.embed_dims)

        self.mlp = MLPBlock(**self.mlp_cfg,act=nn.ReLU)
        self.norm3 = nn.LayerNorm(self.embed_dims)

        self.norm4 = nn.LayerNorm(self.embed_dims)
        self.cross_attn_image_to_token = Attention(**self.cross_attn_cfg)

        if self.is_last:
            self.final_attn_token_to_image = Attention(**self.cross_attn_cfg)
            self.norm_final_attn = nn.LayerNorm(self.embed_dims)



    def forward(self,
                query: Tensor,
                key: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                is_first: bool = False,) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """

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
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=key)
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