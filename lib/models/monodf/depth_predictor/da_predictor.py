"""Depth Anything V2 powered drop-in replacement for ``DepthPredictor``.

This module preserves the exact external interface of MonoDGP's
``DepthPredictor`` so that the rest of the pipeline (RegionSegHead, det2d/
det3d transformers, criterion) remains untouched. The only change is that the
geometry features fed into ``depth_head`` / ``depth_classifier`` now come from
a frozen Depth Anything V2 backbone instead of a tri-source CNN fusion.

The wrapper supports three modes (selected by ``cfg['da_predictor_mode']``):
    * ``'handcrafted'``: bypass DA entirely; behave exactly like MonoDGP
      ``DepthPredictor`` (for ablation / sanity).
    * ``'da_only'``: use DA features alone after a 1x1 projection to
      ``hidden_dim``.
    * ``'hybrid'`` (default for safety): average the DA features with the
      MonoDGP tri-source fusion. With a learnable gate this can fall back to
      pure MonoDGP if DA features hurt.
"""

import os
import sys
from contextlib import nullcontext
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_predictor import DepthPredictor
from .transformer import TransformerEncoder, TransformerEncoderLayer


_DA_FEATURE_CHANNELS = {
    'vits': 64,
    'vitb': 128,
    'vitl': 256,
    'vitg': 384,
}

_DA_OUT_CHANNELS = {
    'vits': [48, 96, 192, 384],
    'vitb': [96, 192, 384, 768],
    'vitl': [256, 512, 1024, 1024],
    'vitg': [1536, 1536, 1536, 1536],
}


def _ensure_da_v2_on_path() -> None:
    """Add the bundled Depth-Anything-V2 source tree to ``sys.path``."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    da_dir = os.path.join(project_root, 'third_party', 'Depth-Anything-V2')
    if os.path.isdir(da_dir) and da_dir not in sys.path:
        sys.path.insert(0, da_dir)


def build_depth_anything_v2(encoder: str = 'vitb', pretrained_path: str = None) -> nn.Module:
    """Instantiate Depth Anything V2 and (optionally) load weights."""
    _ensure_da_v2_on_path()
    from depth_anything_v2.dpt import DepthAnythingV2

    feat = _DA_FEATURE_CHANNELS[encoder]
    out_channels = _DA_OUT_CHANNELS[encoder]
    model = DepthAnythingV2(encoder=encoder, features=feat, out_channels=out_channels)

    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model


class _DAPath1Extractor(nn.Module):
    """Run DA V2 forward, returning both depth and the path_1 geometry features.

    The DPT head's ``path_1`` is a multi-scale fused refinement output with
    ``features`` channels (128 for ViT-B). It encodes the geometry at the
    finest resolution prior to the per-pixel depth conv, and is the same
    feature the MonoDF paper extracts as :math:`G_{map}`.
    """

    def __init__(self, da_model: nn.Module):
        super().__init__()
        self.model = da_model
        self.encoder_name = da_model.encoder

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = x.shape[-2:]
        patch_h, patch_w = H // 14, W // 14

        layer_idx = self.model.intermediate_layer_idx[self.encoder_name]
        feats = self.model.pretrained.get_intermediate_layers(x, layer_idx, return_class_token=True)

        head = self.model.depth_head
        out = []
        for i, item in enumerate(feats):
            if head.use_clstoken:
                tok, cls_tok = item[0], item[1]
                readout = cls_tok.unsqueeze(1).expand_as(tok)
                tok = head.readout_projects[i](torch.cat((tok, readout), -1))
            else:
                tok = item[0]
            tok = tok.permute(0, 2, 1).reshape(tok.shape[0], tok.shape[-1], patch_h, patch_w)
            tok = head.projects[i](tok)
            tok = head.resize_layers[i](tok)
            out.append(tok)
        l1, l2, l3, l4 = out

        l1_rn = head.scratch.layer1_rn(l1)
        l2_rn = head.scratch.layer2_rn(l2)
        l3_rn = head.scratch.layer3_rn(l3)
        l4_rn = head.scratch.layer4_rn(l4)

        path_4 = head.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        path_3 = head.scratch.refinenet3(path_4, l3_rn, size=l2_rn.shape[2:])
        path_2 = head.scratch.refinenet2(path_3, l2_rn, size=l1_rn.shape[2:])
        path_1 = head.scratch.refinenet1(path_2, l1_rn)  # (B, features, ~H/2, ~W/2)

        d = head.scratch.output_conv1(path_1)
        d = F.interpolate(d, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        d = head.scratch.output_conv2(d)
        depth = F.relu(d).squeeze(1)
        return depth, path_1


class DepthAnythingPredictor(nn.Module):
    """Drop-in replacement for ``DepthPredictor`` that injects DA V2 features.

    Returns the same ``(depth_logits, depth_embed, weighted_depth)`` triple as
    MonoDGP's ``DepthPredictor`` so the criterion / decoder pipeline does not
    need to change. The DA forward pass is wrapped in ``torch.no_grad`` when
    the whole DA model is frozen, which keeps the extra cost reasonable.
    """

    def __init__(self, model_cfg):
        super().__init__()
        depth_num_bins = int(model_cfg['num_depth_bins'])
        depth_min = float(model_cfg['depth_min'])
        depth_max = float(model_cfg['depth_max'])
        self.depth_max = depth_max
        d_model = int(model_cfg['hidden_dim'])

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        # ---- DA V2 ---------------------------------------------------------
        encoder = model_cfg.get('depth_anything_encoder', 'vitb')
        pretrained_path = model_cfg.get('depth_anything_pretrained', None)
        self.da_encoder_name = encoder
        self.da_feat_channels = _DA_FEATURE_CHANNELS[encoder]

        da_model = build_depth_anything_v2(encoder=encoder, pretrained_path=pretrained_path)
        self.da_extractor = _DAPath1Extractor(da_model)

        # freeze options
        freeze_full = bool(model_cfg.get('freeze_depth_anything_model', True))
        if freeze_full:
            for p in self.da_extractor.parameters():
                p.requires_grad = False
        else:
            # encoder always frozen by default; DPT head trainable
            for p in self.da_extractor.model.pretrained.parameters():
                p.requires_grad = False
        self._da_fully_frozen = freeze_full

        # ---- DA -> hidden_dim projection ----------------------------------
        self.da_proj = nn.Sequential(
            nn.Conv2d(self.da_feat_channels, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )

        # ---- MonoDGP-style fusion of CNN srcs (kept for hybrid mode) ------
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model))

        # ---- shared head / classifier / encoder (identical to MonoDGP) ----
        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())
        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=1)

        depth_encoder_layer = TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=256, dropout=0.1)
        self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)
        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, d_model)

        # ---- mode + DA input handling -------------------------------------
        self.mode = model_cfg.get('da_predictor_mode', 'hybrid')
        assert self.mode in ('handcrafted', 'da_only', 'hybrid'), self.mode

        # Optional resize of input to a smaller DA-friendly size to bound cost.
        ds = model_cfg.get('depth_anything_input_size', None)
        if ds is None:
            self.da_input_hw = None
        elif isinstance(ds, int):
            self.da_input_hw = (int(ds), int(ds))
        else:
            self.da_input_hw = (int(ds[0]), int(ds[1]))

        # ImageNet normalisation parameters (DA was trained on these)
        self.register_buffer('_da_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_da_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Hybrid-mode learnable gate. Defaults to a strong prior on the CNN
        # path (sigmoid(-2) ~ 0.12, i.e. DA contributes ~12% at init) so that
        # turning DA on cannot dramatically perturb the well-trained MonoDGP
        # detector at the first step. The bias is configurable via
        # ``hybrid_gate_init`` (logit space).
        gate_init = float(model_cfg.get('hybrid_gate_init', -2.0))
        self.hybrid_gate = nn.Parameter(torch.tensor([gate_init]))

        # When True, ``forward`` returns an extra element ``geometry_feat``
        # (the projected DA path_1 at target resolution, before any blending
        # with the CNN tri-source). Used by Stage C-1 OGM as a *pure*
        # geometry memory bank for object queries.
        self.enable_geometry_output = bool(model_cfg.get('enable_geometry_output', False))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _normalise_for_da(self, images: torch.Tensor) -> torch.Tensor:
        """KITTI dataloader returns 0-1 ImageNet-normalised images already.

        Whether DA needs further normalisation depends on the dataset; this
        repository's ``KITTI_Dataset`` already does ``(x - mean) / std`` with
        ImageNet stats, so we simply forward the tensor.
        """
        return images

    def _resize_for_da(self, images: torch.Tensor) -> torch.Tensor:
        if self.da_input_hw is None:
            return images
        H, W = images.shape[-2:]
        if H * W <= self.da_input_hw[0] * self.da_input_hw[1]:
            return images
        return F.interpolate(images, size=self.da_input_hw, mode='bilinear', align_corners=False)

    def _run_da(self, images: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """Forward DA V2 and return (B, da_feat_channels, *target_hw)."""
        x = self._normalise_for_da(images)
        x = self._resize_for_da(x)

        # DA's DINOv2 patch size is 14 -> require multiples of 14
        H, W = x.shape[-2:]
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        ctx = torch.no_grad() if self._da_fully_frozen else nullcontext()
        with ctx:
            _, path1 = self.da_extractor(x)
        if self._da_fully_frozen:
            path1 = path1.detach()

        if path1.shape[-2:] != target_hw:
            path1 = F.interpolate(path1, size=target_hw, mode='bilinear', align_corners=False)
        return path1

    def _fuse_cnn_sources(self, feature: List[torch.Tensor]) -> torch.Tensor:
        """Replicate MonoDGP's tri-source fusion at 1/16 resolution."""
        src_16 = self.proj(feature[1])
        src_32 = self.upsample(F.interpolate(feature[2], size=src_16.shape[-2:], mode='bilinear'))
        src_8 = self.downsample(feature[0])
        return (src_8 + src_16 + src_32) / 3

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, feature, mask, pos, images=None):
        target_hw = feature[1].shape[-2:]

        da_src = None  # populated in da_only / hybrid modes
        if self.mode == 'handcrafted' or images is None:
            src = self._fuse_cnn_sources(feature)
        elif self.mode == 'da_only':
            da_feat = self._run_da(images, target_hw)
            da_src = self.da_proj(da_feat)
            src = da_src
        else:  # hybrid
            cnn_src = self._fuse_cnn_sources(feature)
            da_src = self.da_proj(self._run_da(images, target_hw))
            alpha = torch.sigmoid(self.hybrid_gate)
            src = (1.0 - alpha) * cnn_src + alpha * da_src

        src = self.depth_head(src)
        depth_logits = self.depth_classifier(src)

        depth_probs = F.softmax(depth_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)

        B, C, H, W = src.shape
        src_flat = src.flatten(2).permute(2, 0, 1)
        mask_flat = mask.flatten(1)
        pos_flat = pos.flatten(2).permute(2, 0, 1)

        depth_embed = self.depth_encoder(src_flat, mask_flat, pos_flat)
        depth_embed = depth_embed.permute(1, 2, 0).reshape(B, C, H, W)

        if self.enable_geometry_output:
            # Fall back to ``src`` when running in handcrafted mode so the
            # caller always gets a tensor with shape (B, C, H, W).
            geometry_feat = da_src if da_src is not None else src
            return depth_logits, depth_embed, weighted_depth, geometry_feat

        return depth_logits, depth_embed, weighted_depth


def build_depth_predictor(model_cfg) -> nn.Module:
    """Factory returning either MonoDGP's ``DepthPredictor`` or the DA wrapper.

    Selection key (in ``model_cfg``):
        * ``use_depth_anything: False`` (default) -> handcrafted DepthPredictor
        * ``use_depth_anything: True``            -> ``DepthAnythingPredictor``
    """
    if bool(model_cfg.get('use_depth_anything', False)):
        return DepthAnythingPredictor(model_cfg)
    return DepthPredictor(model_cfg)
