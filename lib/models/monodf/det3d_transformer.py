from typing import Optional, List
import math
import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from utils.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn, MSDeformAttn_cross, MultiheadAttention


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Det3DTransformer(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            return_intermediate_dec=False,
            num_feature_levels=4,
            dec_n_points=4,
            group_num=1,
            use_ogm=False,
            ogm_gate_init=-3.0,
            ogm_gate_cap=1.0,
            ogm_gate_schedule=None,
            use_gqr=False,
            gqr_K=10,
            gqr_gate_init=-3.0,
            gqr_gate_cap=1.0,
            gqr_gate_schedule=None):

        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.group_num = group_num
        self.use_ogm = bool(use_ogm)
        self.use_gqr = bool(use_gqr)
        self.ogm_gate_schedule = ogm_gate_schedule
        self.gqr_gate_schedule = gqr_gate_schedule

        decoder_layer = DepthAwareDecoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points,
            group_num=group_num, use_ogm=self.use_ogm, ogm_gate_init=ogm_gate_init,
            ogm_gate_cap=ogm_gate_cap)
        self.decoder = DepthAwareDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec,
                                         use_gqr=self.use_gqr, d_model=d_model,
                                         gqr_K=gqr_K, gqr_gate_init=gqr_gate_init,
                                         gqr_gate_cap=gqr_gate_cap)

        self._reset_parameters()
        # Re-zero the OGM output projections after the global xavier init.
        # This way the OGM residual is *exactly* zero at init regardless of
        # the gate value, so loading a Stage B ckpt + dropping in OGM yields
        # bit-identical predictions on the first forward.
        if self.use_ogm:
            for layer in self.decoder.layers:
                if hasattr(layer, 'geo_cross_attn'):
                    nn.init.zeros_(layer.geo_cross_attn.out_proj.weight)
                    nn.init.zeros_(layer.geo_cross_attn.out_proj.bias)
        # GQR: zero-init the projection from Fourier features to d_model so
        # that the residual injected between decoder layers is exactly zero
        # at init. With ``LayerNorm(0) = 0`` (default beta=0), the post-norm
        # delta is also zero and Stage C-2 reproduces Stage B bit-for-bit at
        # the first forward, regardless of the gate value or the per-query
        # depth predictor's output.
        if self.use_gqr:
            nn.init.zeros_(self.decoder.gqr_proj[0].weight)
            nn.init.zeros_(self.decoder.gqr_proj[0].bias)

    @staticmethod
    def _scheduled_cap(base_cap, schedule, epoch):
        if schedule is None or epoch is None:
            return float(base_cap)
        if not schedule:
            return float(base_cap)
        points = sorted((int(p[0]), float(p[1])) for p in schedule)
        if epoch <= points[0][0]:
            return points[0][1]
        for (e0, c0), (e1, c1) in zip(points[:-1], points[1:]):
            if epoch <= e1:
                ratio = float(epoch - e0) / max(float(e1 - e0), 1.0)
                return c0 + ratio * (c1 - c0)
        return points[-1][1]

    def set_epoch(self, epoch):
        if self.use_ogm:
            cap = self._scheduled_cap(
                self.decoder.layers[0].ogm_gate_base_cap,
                self.ogm_gate_schedule,
                epoch)
            for layer in self.decoder.layers:
                if hasattr(layer, 'ogm_gate_cap'):
                    layer.ogm_gate_cap = cap
        if self.use_gqr:
            self.decoder.gqr_gate_cap = self._scheduled_cap(
                self.decoder.gqr_gate_base_cap,
                self.gqr_gate_schedule,
                epoch)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()


    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, intermediate_output, query_embeds, depth_pos_embed,
                geometry_memory=None, geometry_pos=None, geometry_mask=None):

        # prepare input for decoder
        memory = intermediate_output['memory']
        reference_points = intermediate_output['reference_points']
        spatial_shapes = intermediate_output['spatial_shapes']
        level_start_index = intermediate_output['level_start_index']
        valid_ratios = intermediate_output['valid_ratios']
        mask_flatten = intermediate_output['mask_flatten']
        
        bs, _, c = memory.shape
        tgt = query_embeds #intermediate_output['hs'][-1]
        init_reference_out = reference_points

        depth_pos_embed = depth_pos_embed.flatten(2).permute(2, 0, 1)
        mask_depth = None
        query_embeds = None

        # OGM memory tokens: (B, C, Hg, Wg) -> (Hg*Wg, B, C). Pos embedding
        # has the same shape; mask is (B, Hg, Wg) -> (B, Hg*Wg).
        geo_mem_flat = geo_pos_flat = geo_mask_flat = None
        if self.use_ogm and geometry_memory is not None:
            geo_mem_flat = geometry_memory.flatten(2).permute(2, 0, 1)
            if geometry_pos is not None:
                geo_pos_flat = geometry_pos.flatten(2).permute(2, 0, 1)
            if geometry_mask is not None:
                geo_mask_flat = geometry_mask.flatten(1)

        # decoder
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embeds,
            mask_flatten,
            depth_pos_embed,
            mask_depth,
            bs=bs,
            geo_memory=geo_mem_flat,
            geo_pos=geo_pos_flat,
            geo_mask=geo_mask_flat)

        inter_references_out = inter_references

        return hs, init_reference_out, inter_references_out


class DepthAwareDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, group_num=1,
                 use_ogm=False, ogm_gate_init=-3.0, ogm_gate_cap=1.0):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # depth cross attention
        self.cross_attn_depth = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_depth = nn.Dropout(dropout)
        self.norm_depth = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.group_num = group_num

        # ---- Stage C-1: Object-centric Geometry Memory (OGM) ---------------
        # A standard MHA cross-attention from object queries to *pure* DA
        # geometry tokens, with a learnable scalar gate. The output projection
        # is zero-initialised (see Det3DTransformer._reset_parameters override)
        # so that at init time this layer contributes exactly zero, and the
        # pre-trained Stage B ckpt is preserved bit-for-bit.
        self.use_ogm = bool(use_ogm)
        self.ogm_gate_base_cap = float(ogm_gate_cap)
        self.ogm_gate_cap = float(ogm_gate_cap)
        if self.use_ogm:
            self.geo_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.geo_dropout = nn.Dropout(dropout)
            self.geo_norm = nn.LayerNorm(d_model)
            self.geo_gate = nn.Parameter(torch.tensor([float(ogm_gate_init)]))
        
        
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                query_pos,
                reference_points,
                src,
                src_spatial_shapes,
                level_start_index,
                src_padding_mask,
                depth_pos_embed,
                mask_depth,
                bs,
                geo_memory=None,
                geo_pos=None,
                geo_mask=None):

        # depth cross attention
        tgt2 = self.cross_attn_depth(tgt.transpose(0, 1),
                                     depth_pos_embed,
                                     depth_pos_embed,
                                     key_padding_mask=mask_depth)[0].transpose(0, 1)
       
        tgt = tgt + self.dropout_depth(tgt2)
        tgt = self.norm_depth(tgt)

        # Stage C-1: Object-centric Geometry Memory cross-attention. Queries
        # attend to a pure-DA dense geometry bank.
        # We use a "delta-gating" pattern: normalise the OGM output (so its
        # scale stays controlled regardless of how the gate grows), then add
        # ``gate * delta`` to the original ``tgt``. With out_proj zero-init
        # *and* geo_norm bias zero-init (default), delta is identically zero
        # at init, so the residual is exactly zero -> the layer is bit-
        # identical to the no-OGM baseline. As the gate / out_proj move,
        # delta becomes non-trivial without destabilising the rest of the
        # decoder which was trained on LayerNorm-scaled features.
        if self.use_ogm and geo_memory is not None:
            q_in = self.with_pos_embed(tgt, query_pos).transpose(0, 1)
            k_in = self.with_pos_embed(geo_memory, geo_pos) if geo_pos is not None else geo_memory
            tgt2 = self.geo_cross_attn(q_in,
                                       k_in,
                                       geo_memory,
                                       key_padding_mask=geo_mask)[0].transpose(0, 1)
            gate = self.ogm_gate_cap * torch.sigmoid(self.geo_gate)
            delta = self.geo_norm(self.geo_dropout(tgt2))
            tgt = tgt + gate * delta

        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = tgt.transpose(0, 1)
        num_queries = q.shape[0]
       
        if self.training:
            num_noise = num_queries-self.group_num * 50
            num_queries = self.group_num * 50
            q_noise = q[:num_noise].repeat(1,self.group_num, 1)
            k_noise = k[:num_noise].repeat(1,self.group_num, 1)
            v_noise = v[:num_noise].repeat(1,self.group_num, 1)
            q = q[num_noise:]
            k = k[num_noise:]
            v = v[num_noise:]
            q = torch.cat(q.split(num_queries // self.group_num, dim=0), dim=1)
            k = torch.cat(k.split(num_queries // self.group_num, dim=0), dim=1)
            v = torch.cat(v.split(num_queries // self.group_num, dim=0), dim=1)
            q = torch.cat([q_noise,q], dim=0)
            k = torch.cat([k_noise,k], dim=0)
            v = torch.cat([v_noise,v], dim=0)
        
        tgt2 = self.self_attn(q, k, v)[0]
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=1), dim=0).transpose(0, 1)
            
        else:
            tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
      
        
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DepthAwareDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False,
                 use_gqr=False, d_model=256, gqr_K=10, gqr_gate_init=-3.0,
                 gqr_gate_cap=1.0):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.dim_embed = None
        self.class_embed = None

        # ---- Stage C-2: Geometry-aware Query Refinement (GQR) -------------
        # Each decoder layer (except the first) receives an additive residual
        # encoding the previous layer's per-query depth state. The depth is
        # produced by a small head ``gqr_depth_head: Linear(d_model, 1)``
        # squashed by sigmoid into [0, 1]. It is then expanded into a 2K-dim
        # Fourier basis and projected back to ``d_model`` with a learnable
        # MLP whose weights are zero-initialised by ``Det3DTransformer`` to
        # guarantee identity at init.
        self.use_gqr = bool(use_gqr)
        self.gqr_gate_base_cap = float(gqr_gate_cap)
        self.gqr_gate_cap = float(gqr_gate_cap)
        if self.use_gqr:
            self.gqr_K = int(gqr_K)
            self.gqr_depth_head = nn.Linear(d_model, 1)
            self.gqr_proj = nn.Sequential(
                nn.Linear(2 * self.gqr_K, d_model),
                nn.LayerNorm(d_model),
            )
            self.gqr_gate = nn.Parameter(torch.tensor([float(gqr_gate_init)]))
            # Frequencies (NeRF-style log-spaced).
            freqs = (2.0 ** torch.arange(self.gqr_K, dtype=torch.float32)) * math.pi
            self.register_buffer('gqr_freqs', freqs)

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, depth_pos_embed=None, mask_depth=None, bs=None,
                geo_memory=None, geo_pos=None, geo_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        bs = src.shape[0]
        
        for lid, layer in enumerate(self.layers):

            # GQR: inject a Fourier-encoded depth state derived from the
            # previous layer's output before processing the current layer.
            # At init, ``gqr_proj[0]`` weight & bias are zero so ``delta`` is
            # zero regardless of the gate or the depth-head prediction; this
            # makes Stage C-2 bit-identical to Stage B on the first forward.
            if self.use_gqr and lid > 0:
                d_norm = torch.sigmoid(self.gqr_depth_head(output))  # (B, Q, 1)
                phases = d_norm * self.gqr_freqs.view(1, 1, -1)       # (B, Q, K)
                fourier = torch.cat([phases.sin(), phases.cos()], dim=-1)  # (B, Q, 2K)
                delta = self.gqr_proj(fourier)                        # (B, Q, d_model)
                gate = self.gqr_gate_cap * torch.sigmoid(self.gqr_gate)
                output = output + gate * delta

            if reference_points.shape[-1] == 6:
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
                
            output = layer(output,
                           query_pos,
                           reference_points_input,
                           src,
                           src_spatial_shapes,
                           src_level_start_index,
                           src_padding_mask,
                           depth_pos_embed,
                           mask_depth,
                           bs,
                           geo_memory=geo_memory,
                           geo_pos=geo_pos,
                           geo_mask=geo_mask)

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_det3d_transformer(cfg):
    return Det3DTransformer(
        d_model=cfg['hidden_dim'],
        dropout=cfg['dropout'],
        activation="relu",
        nhead=cfg['nheads'],
        dim_feedforward=cfg['dim_feedforward'],
        num_decoder_layers=cfg['dec_layers'],
        return_intermediate_dec=cfg['return_intermediate_dec'],
        num_feature_levels=cfg['num_feature_levels'],
        dec_n_points=cfg['dec_n_points'],
        use_ogm=cfg.get('use_ogm', False),
        ogm_gate_init=cfg.get('ogm_gate_init', -3.0),
        ogm_gate_cap=cfg.get('ogm_gate_cap', 1.0),
        ogm_gate_schedule=cfg.get('ogm_gate_schedule', None),
        use_gqr=cfg.get('use_gqr', False),
        gqr_K=cfg.get('gqr_K', 10),
        gqr_gate_init=cfg.get('gqr_gate_init', -3.0),
        gqr_gate_cap=cfg.get('gqr_gate_cap', 1.0),
        gqr_gate_schedule=cfg.get('gqr_gate_schedule', None))
