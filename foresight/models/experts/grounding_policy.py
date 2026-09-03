from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet
from depth_anything_v2.dinov2 import DINOv2


from cotnav.models.adapters import replace_bn_with_gn, LambdaLayer
from cotnav.models.blocks.convnet_spatial import ConvNetSpatial
from cotnav.models.blocks.dense_network import DenseNetwork
from cotnav.models.blocks.transformer_head import TransformerActionHead

class SpatialGroundingPolicy(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        
        self.obs_cfg = cfg["obs_encoder"]
        self.path_cfg = cfg["path_encoder"]
        self.spatial_cfg = cfg["spatial_encoder"]
        self.action_cfg = cfg["action_head"]

        #1 Initialize depth anything encoder
        obs_model_name = self.obs_cfg["name"]
        self.obs_encoder = DINOv2(model_name=obs_model_name)
        for param in self.obs_encoder.parameters():
            param.requires_grad = False
        self.obs_layer_idx = self.obs_cfg["dino_layer_idx"][obs_model_name]
        self.obs_pool_dim = self.obs_cfg["pool_dim"]
        self.obs_enc_dim = self.obs_cfg["out_dim"][obs_model_name]
        self.num_obs_features = self.obs_enc_dim * self.obs_pool_dim

        #2 Initialize effnet waypoint encoder 
        self.path_encoder = replace_bn_with_gn(EfficientNet.from_name(self.path_cfg["name"], in_channels=self.path_cfg["in_channels"]))
        # We only use extract_features(...), so EfficientNet classifier params are unused.
        if hasattr(self.path_encoder, "_fc"):
            for param in self.path_encoder._fc.parameters():
                param.requires_grad = False
        path_enc_dim = self.path_encoder._conv_head.out_channels
        path_proj_dim = self.path_cfg["out_dim"]
        self.path_proj = nn.Sequential(
            nn.Conv2d(path_enc_dim, path_proj_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32 if path_enc_dim % 32 == 0 else 16, num_channels=path_proj_dim),
            nn.SiLU(inplace=True),
        )
        #3 Initialize spatial softmax
        self.spatial_softmax = globals()[self.spatial_cfg['name']](self.spatial_cfg)

        #4 Initialize mlp head
        self.mlp_head = globals()[self.action_cfg['name']](**self.action_cfg['kwargs'])


    def forward(self, inputs: dict) -> dict:
        obs_img = inputs["rgb"]
        path_mask = inputs["path_mask"]

        obs_feats_all = self.obs_encoder.get_intermediate_layers(
            obs_img, self.obs_layer_idx, reshape=True, return_class_token=False
        )
        obs_feats_last = obs_feats_all[-1] # [B, F, H, W]

        path_feats = self.path_proj(self.path_encoder.extract_features(path_mask))
        Hp, Wp = path_feats.shape[2:]
        Hobs, Wobs = obs_feats_last.shape[2:]
        if Hp != Hobs or Wp != Wobs:
            path_feats = F.interpolate(path_feats, size=(Hobs, Wobs), mode='bilinear', align_corners=False)

        spatial_feats = self.spatial_softmax({
            "obs_feats": obs_feats_last,
            "path_feats": path_feats,
        })[self.spatial_cfg['encoder']['out_key']]
        action_pred = self.mlp_head(spatial_feats) # [B, num_actions, action_dim]

        return {
            "action_pred": action_pred,
        }

if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/model/waypoint/gtpassthrough_simple.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SpatialGroundingPolicy(cfg['waypoint_policy']).to(device)
    inputs = {
        "rgb": torch.randn(1, 3, 224, 294).to(device),
        "path_mask": torch.randn(1, 1, 224, 294).to(device),
    }
    outputs = policy(inputs)
    print(outputs)