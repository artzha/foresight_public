import torch
import torch.nn as nn


class TransformerActionHead(nn.Module):
    """
    Option-C transformer head: trajectory queries + keypoint tokens attend jointly
    via a single self-attention stack. Queries are read out and projected to actions.
    """

    def __init__(
        self,
        num_kp: int,
        kp_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_actions: int,
        action_dim: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        predict_deltas: bool = False,
    ):
        super().__init__()
        self.num_kp = num_kp
        self.kp_dim = kp_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.predict_deltas = predict_deltas

        self.token_proj = nn.Linear(kp_dim, d_model)

        # Per-slot learnable embeddings; keypoints have stable slot identity from spatial softmax
        self.kp_pos_embed = nn.Parameter(torch.zeros(1, num_kp, d_model))
        # One learnable query per waypoint timestep
        self.wp_queries = nn.Parameter(torch.zeros(1, num_actions, d_model))
        self.wp_pos_embed = nn.Parameter(torch.zeros(1, num_actions, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LN is more stable for shallow stacks trained from scratch
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, action_dim)

        nn.init.normal_(self.kp_pos_embed, std=0.02)
        nn.init.normal_(self.wp_queries, std=0.02)
        nn.init.normal_(self.wp_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_kp * kp_dim)
        B = x.shape[0]

        tokens = x.view(B, self.num_kp, self.kp_dim)      # (B, K, kp_dim)
        tokens = self.token_proj(tokens) + self.kp_pos_embed  # (B, K, d_model)

        queries = self.wp_queries.expand(B, -1, -1) + self.wp_pos_embed  # (B, T, d_model)

        # Queries first so we can slice them cleanly after attention
        seq = torch.cat([queries, tokens], dim=1)          # (B, T+K, d_model)
        seq = self.encoder(seq)

        wp_out = self.norm(seq[:, : self.num_actions])     # (B, T, d_model)
        action_pred = self.head(wp_out)                    # (B, T, action_dim)

        if self.predict_deltas:
            action_pred = action_pred.cumsum(dim=1)

        return action_pred
