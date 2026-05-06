import math
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling import ImageEncoder


class LoraLinear(nn.Module):
    """Standard LoRA wrapper around a frozen linear layer."""

    def __init__(self, original_linear: nn.Linear, rank=8, alpha=1.0):
        super().__init__()
        self.size_in = original_linear.in_features
        self.size_out = original_linear.out_features
        self.rank = rank
        self.alpha = alpha

        self.mlp = original_linear
        self.mlp.weight.requires_grad = False
        if self.mlp.bias is not None:
            self.mlp.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, self.size_in))
        self.lora_B = nn.Parameter(torch.zeros(self.size_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = self.alpha / self.rank

    def forward(self, x):
        return self.mlp(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling

    def calculate_delta_w(self):
        with torch.no_grad():
            return (self.lora_B @ self.lora_A) * self.scaling

    def calculate_orthogonality_loss(self):
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        rows, cols = delta_W.shape
        if rows < cols:
            mat = delta_W @ delta_W.T
            identity = torch.eye(rows, device=delta_W.device)
        else:
            mat = delta_W.T @ delta_W
            identity = torch.eye(cols, device=delta_W.device)
        return torch.norm(mat - identity, p="fro")


class LoraATTImageEncoder(ImageEncoder):
    """ImageEncoder with LoRA applied to attention QKV and output projections."""

    def __init__(self, args, keep_lang=False):
        super().__init__(args, keep_lang=keep_lang)
        self.args = args
        rank = args.lora_rank
        alpha = args.lora_alpha
        self.model.visual = self._replace_attention_layers(self.model.visual, rank, alpha)
        self._freeze_non_lora_params()
        self._print_trainable_param_count()

    def _replace_attention_layers(self, vit_model, rank, alpha):
        def new_attn_forward(self, query, key, value, need_weights=False, attn_mask=None, average_attn_weights=True):
            is_batched = query.dim() == 3
            if self.batch_first and is_batched:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

            qkv = self.lora_in_proj_layer(query)

            tgt_len, bsz, embed_dim = query.shape
            qkv = qkv.view(tgt_len, bsz, 3, self.num_heads, self.head_dim).permute(2, 1, 3, 0, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
            attn_output = self.out_proj(attn_output)

            if self.batch_first and is_batched:
                attn_output = attn_output.transpose(1, 0)
            return attn_output, None

        print(f"Replacing attention layers with LoRA (rank={rank}, alpha={alpha})...")
        for block in vit_model.transformer.resblocks:
            attn = block.attn
            embed_dim = attn.embed_dim

            dummy_in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=attn.in_proj_bias is not None)
            dummy_in_proj.weight.data.copy_(attn.in_proj_weight)
            if attn.in_proj_bias is not None:
                dummy_in_proj.bias.data.copy_(attn.in_proj_bias)
            attn.lora_in_proj_layer = LoraLinear(dummy_in_proj, rank=rank, alpha=alpha)
            del attn.in_proj_weight
            if hasattr(attn, "in_proj_bias") and attn.in_proj_bias is not None:
                del attn.in_proj_bias

            attn.out_proj = LoraLinear(attn.out_proj, rank=rank, alpha=alpha)
            attn.forward = types.MethodType(new_attn_forward, attn)

        print("Attention layer replacement complete.")
        return vit_model

    def _freeze_non_lora_params(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True

    def _print_trainable_param_count(self):
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {count:,}")

    def get_delta_w_dict(self):
        """Extract delta_W for each LoRA module, mapped to original weight keys."""
        delta_w_dict = {}
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if not isinstance(module, LoraLinear):
                    continue
                delta_W = module.calculate_delta_w()
                if "lora_in_proj_layer" in name or "in_proj_layer" in name:
                    attn_pos = name.rfind(".attn.")
                    if attn_pos != -1:
                        base = name[:attn_pos]
                        key = f"model.{base}.attn.in_proj_weight"
                    else:
                        key = f"model.{name}.weight"
                else:
                    key = f"model.{name}.weight"
                delta_w_dict[key] = delta_W
        return delta_w_dict

    def calculate_total_orthogonality_loss(self):
        total = 0.0
        for module in self.model.modules():
            if isinstance(module, LoraLinear):
                total = total + module.calculate_orthogonality_loss()
        return total

    def save(self, filename):
        print(f"Saving LoraATTImageEncoder to {filename}")
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self.state_dict(), filename)

    @classmethod
    def load(cls, filename, args):
        print(f"Loading LoraATTImageEncoder from {filename}")
        encoder = cls(args)
        state_dict = torch.load(filename, map_location="cpu")
        encoder.load_state_dict(state_dict)
        return encoder
