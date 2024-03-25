import torch
from torch import nn
import einops
import numpy as np
import math

# from .mhsa import MultiHeadSelfAttention
#
#
# def get_module_device(parameter: nn.Module):
#     try:
#         return next(parameter.parameters()).device
#     except StopIteration:
#         # For nn.DataParallel compatibility in PyTorch 1.5
#
#         def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
#             tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
#             return tuples
#
#         gen = parameter._named_members(get_members_fn=find_tensor_attributes)
#         first_tuple = next(gen)
#         return first_tuple[1].device
#
#
# class TransformerBlock(nn.Module):
#     """
#     Vanilla transformer block from the original paper "Attention is all you need"
#     Detailed analysis: https://theaisummer.com/transformer/
#     """
#
#     def __init__(self, dim, heads=8, dim_head=None,
#                  dim_linear_block=1024, dropout=0.1, activation=nn.ReLU,
#                  mhsa=None, prenorm=False):
#         """
#         Args:
#             dim: token's vector length
#             heads: number of heads
#             dim_head: if none dim/heads is used
#             dim_linear_block: the inner projection dim
#             dropout: probability of droppping values
#             mhsa: if provided you can change the vanilla self-attention block
#             prenorm: if the layer norm will be applied before the mhsa or after
#         """
#         super().__init__()
#         self.mhsa = mhsa if mhsa is not None else MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
#         self.prenorm = prenorm
#         self.drop = nn.Dropout(dropout)
#         self.norm_1 = nn.LayerNorm(dim)
#         self.norm_2 = nn.LayerNorm(dim)
#
#         self.linear = nn.Sequential(
#             nn.Linear(dim, dim_linear_block),
#             activation(),  # nn.ReLU or nn.GELU
#             nn.Dropout(dropout),
#             nn.Linear(dim_linear_block, dim)
#         )
#
#     def forward(self, x, mask=None):
#         if self.prenorm:
#             y = self.drop(self.mhsa(self.norm_1(x), mask)) + x
#             out = self.linear(self.norm_2(y)) + y
#         else:
#             y = self.norm_1(self.drop(self.mhsa(x, mask)) + x)
#             out = self.norm_2(self.linear(y) + y)
#         return out
#
#
# def weight_init(m):
#     if isinstance(m, nn.Linear):
#         m.weight.data.zero_()
#         if m.bias is not None:
#             m.bias.data.zero_()

from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn.functional as F


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=1, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttentionSpatial(nn.Module):
    def __init__(
            self, d_model, n_head, d_head, dropout=0.1, spatial_multihead=False, spatial_dim=5,
            spatial_attn_fusion='mul',
    ):
        super().__init__()
        assert d_model % n_head == 0, 'd_model: %d, n_head: %d' % (d_model, n_head)

        self.n_head = n_head
        self.d_model = d_model
        self.d_per_head = d_head
        d_inner = d_head * n_head
        self.spatial_multihead = spatial_multihead
        self.spatial_dim = spatial_dim
        self.spatial_attn_fusion = spatial_attn_fusion

        self.w_qs = nn.Linear(d_model, d_inner)
        self.w_ks = nn.Linear(d_model, d_inner)
        self.w_vs = nn.Linear(d_model, d_inner)

        self.fc = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(p=dropout)
        # self.layer_norm = nn.LayerNorm(d_model)

        # self.spatial_n_head = n_head if spatial_multihead else 1
        # if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
        #     self.pairwise_loc_fc = nn.Linear(spatial_dim, self.spatial_n_head)
        # elif self.spatial_attn_fusion == 'ctx':
        #     self.pairwise_loc_fc = nn.Linear(spatial_dim, d_model)
        # elif self.spatial_attn_fusion == 'cond':
        #     self.lang_cond_fc = nn.Linear(d_model, self.spatial_n_head * (spatial_dim + 1))
        # else:
        #     raise NotImplementedError('unsupported spatial_attn_fusion %s' % (self.spatial_attn_fusion))

    def forward(self, q, k, v, pairwise_locs, key_padding_mask=None, txt_embeds=None):
        # residual = q
        q = einops.rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = einops.rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
        v = einops.rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])

        # if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
        #     loc_attn = self.pairwise_loc_fc(pairwise_locs)
        #     loc_attn = einops.rearrange(loc_attn, 'b l t h -> h b l t')
        #     # loc_attn = torch.sigmoid(loc_attn)
        #     # if self.spatial_attn_fusion == 'mul':
        #     #     loc_attn = F.relu(loc_attn)
        #     if not self.spatial_multihead:
        #         loc_attn = einops.repeat(loc_attn, 'h b l t -> (h nh) b l t', nh=self.n_head)
        # elif self.spatial_attn_fusion == 'ctx':
        #     loc_attn = self.pairwise_loc_fc(pairwise_locs)
        #     loc_attn = einops.rearrange(loc_attn, 'b l t (h k) -> h b l t k', h=self.n_head)
        #     loc_attn = torch.einsum('hblk,hbltk->hblt', q, loc_attn) / np.sqrt(q.shape[-1])
        # elif self.spatial_attn_fusion == 'cond':
        #     spatial_weights = self.lang_cond_fc(residual + txt_embeds.unsqueeze(1))
        #     spatial_weights = einops.rearrange(spatial_weights, 'b l (h d) -> h b l d', h=self.spatial_n_head,
        #                                        d=self.spatial_dim + 1)
        #     if self.spatial_n_head == 1:
        #         spatial_weights = einops.repeat(spatial_weights, '1 b l d -> h b l d', h=self.n_head)
        #     spatial_bias = spatial_weights[..., :1]
        #     spatial_weights = spatial_weights[..., 1:]
        #     loc_attn = torch.einsum('hbld,bltd->hblt', spatial_weights, pairwise_locs) + spatial_bias
        #     loc_attn = torch.sigmoid(loc_attn)

        # pairwise_dis = pairwise_locs[..., 0]
        # pairwise_dis_norm = pairwise_dis / (pairwise_dis.max(dim=-1, keepdim=True)[0] * 1.5)
        # pairwise_dis_norm[pairwise_dis_norm < 0.3] = 0.
        # pairwise_dis_norm = einops.repeat(pairwise_dis_norm, 'b l t -> h b l t', h=self.n_head)
        # dis_mask = torch.bernoulli(pairwise_dis_norm)

        if key_padding_mask is not None:
            mask = einops.repeat(key_padding_mask, 'b t -> h b l t', h=self.n_head, l=q.size(2))
            attn = attn.masked_fill(mask, -np.inf)
            # if self.spatial_attn_fusion in ['mul', 'cond']:
            #     loc_attn = loc_attn.masked_fill(mask, 0)
            # else:
            # loc_attn = loc_attn.masked_fill(mask, -np.inf)

        # if self.spatial_attn_fusion == 'add':
        #     fused_attn = (torch.softmax(attn, 3) + torch.softmax(loc_attn, 3)) / 2
        # else:
        #     if self.spatial_attn_fusion in ['mul', 'cond']:
        #         fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
        #     else:
        # breakpoint()
        # attn = torch.nn.functional.normalize(attn, dim=-1)
        fused_attn = torch.softmax(attn, -1)
        # if torch.sum(torch.isnan(fused_attn) == 1) > 0:
        assert torch.sum(torch.isnan(fused_attn) == 1) == 0, print(fused_attn)

        output = torch.einsum('hblt,hbtv->hblv', fused_attn, v)
        output = einops.rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        # output = self.layer_norm(output + residual)
        return output


class TransformerSpatialDecoderLayer(nn.Module):
    def __init__(
            self, d_model, nhead, dim_head, dim_feedforward, dropout,
            spatial_multihead=True, spatial_dim=5, spatial_attn_fusion='mul'
    ):
        super().__init__()
        # nhead = d_model // 64
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dim_head, dropout=dropout,
            spatial_multihead=spatial_multihead,
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )
        # self.multihead_attn = nn.MultiheadAttention(
        #     d_model, nhead, dropout=dropout, batch_first=True
        # )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(
            self, tgt, tgt_pairwise_locs,
            tgt_key_padding_mask=None,
    ):
        tgt2 = self.self_attn(
            tgt, tgt, tgt, tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = (tgt + tgt2) / math.sqrt(2)
        # tgt2 = self.norm2(tgt)
        # tgt2, cross_attn_matrices = self.multihead_attn(
        #     query=tgt2, key=memory,
        #     value=memory, attn_mask=memory_mask,
        #     key_padding_mask=memory_key_padding_mask
        # )
        # tgt = tgt + self.dropout2(tgt2)
        # tgt2 = self.norm3(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        # tgt = tgt + self.dropout3(tgt2)
        return tgt


import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CMT(nn.Module):

    def __init__(self, hidden_size, num_layers=1):
        super().__init__()

        decoder_layer = TransformerSpatialDecoderLayer(
            d_model=hidden_size, nhead=8, dim_head=64,
            dim_feedforward=4096, dropout=0.1
        )
        self.layers = _get_clones(decoder_layer, num_layers)

        # loc_layer = nn.Sequential(
        #     nn.Linear(6, hidden_size),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_size)
        # )
        # self.loc_layers = _get_clones(loc_layer, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def calc_pairwise_locs(self, obj_centers, eps=1e-10, pairwise_rel_type='center'):
        pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
                        - einops.repeat(obj_centers, 'b l d -> b 1 l d')
        pairwise_dists = torch.sqrt(torch.sum(pairwise_locs ** 2, 3) + eps)  # (b, l, l)

        # max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
        norm_pairwise_dists = pairwise_dists #/ einops.repeat(max_dists, 'b -> b 1 1')

        pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, 3) + eps)
        pairwise_locs = torch.stack(
            [norm_pairwise_dists, pairwise_locs[..., 2] / pairwise_dists,
             pairwise_dists_2d / pairwise_dists, pairwise_locs[..., 1] / pairwise_dists_2d,
             pairwise_locs[..., 0] / pairwise_dists_2d],
            dim=3
        )
        return pairwise_locs

    def forward(
            self, obj_embeds, obj_locs, obj_masks
    ):
        pairwise_locs = self.calc_pairwise_locs(
            obj_locs[:, :, :3]
        )

        out_embeds = obj_embeds
        for i, layer in enumerate(self.layers):
            # query_pos = self.loc_layers[0](obj_locs)
            # out_embeds = out_embeds + query_pos
            out_embeds = layer(
                out_embeds, pairwise_locs,
                tgt_key_padding_mask=obj_masks.logical_not(),
            )

        return out_embeds


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.gate = torch.nn.Parameter(torch.zeros(self.heads, 1, 1))

    def forward(self, x, context=None, mask=None, dist_attn=None):
        h = self.heads

        def print_grad_hook(name):
            def print_hook(grad):
                if not torch.isnan(grad.mean()):
                    print(f"max abs grad of {name}: {grad.abs().max()}")
                    # breakpoint()

            return print_hook

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # v = rearrange(v, 'b n (h d) -> (b h) n d', h=h)
        # q.register_hook(print_grad_hook("q"))
        # k.register_hook(print_grad_hook("k"))
        # v.register_hook(print_grad_hook("v"))
        # print(f"max abs value of q: {q.abs().max()}")
        # print(f"max abs value of k: {k.abs().max()}")
        # print(f"max abs value of v: {v.abs().max()}")

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # sim.register_hook(print_grad_hook("sim"))

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        # print("sim:", sim[0].max(dim=-1)[0])
        attn = sim.softmax(dim=-1)
        # print("attn:", attn[0].max(dim=-1))
        # print(f"max abs value of attn: {attn.abs().max()}")
        # attn.register_hook(print_grad_hook("attn"))
        out = einsum('b i j, b j d -> b i d', attn, v)
        # print(f"max abs value of out: {out.abs().max()}")
        # out.register_hook(print_grad_hook("out"))
        # breakpoint()
        # dist_attn = repeat(dist_attn, 'b i j -> (b h) i j', h=h)
        # out = einsum('b i j, b j d -> b i d', dist_attn, v)

        if dist_attn is not None:
            dist_attn = repeat(dist_attn, 'b i j -> (b h) i j', h=h)
            dist_attn_out = einsum('b i j, b j d -> b i d', dist_attn, v)
            # gates = self.gate.sigmoid().to(dist_attn.device)
            # gates = gates.repeat(x.shape[0], 1, 1)
            # gates.register_hook(lambda grad: grad * 10.)
            # print(f"max/min abs value of gates: {gates.abs().max()} {gates.abs().min()}")
            # gates.register_hook(print_grad_hook("gates"))
            ratio = 0.1
            out = (1 - ratio) * out + ratio * dist_attn_out
            # print(f"max abs value of out: {out.abs().max()}")
            # out.register_hook(print_grad_hook("out"))

        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=8, d_head=None, dropout=0., context_dim=None, gated_ff=False):
        super().__init__()
        if d_head is None:
            d_head = dim // n_heads
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head,
                                    dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
        #                             heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None, mask=None, dist_attn=None):
        x = self.attn1(self.norm1(x), mask=mask, dist_attn=dist_attn) + x
        # _x = self.attn2(self.norm2(x), context=context, mask=mask)
        # x = _x + x
        x = self.ff(self.norm3(x)) + x
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers=1, heads=32, dim_head=None, dropout=0.1):
        super().__init__()
        self.block_list = [BasicTransformerBlock(dim, heads, dim_head, dropout) for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.block_list)
        self.output_norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def forward(self, x, mask=None, dist_attn=None):
        for layer in self.layers:
            x = layer(x, mask=mask, dist_attn=dist_attn)
        # x = self.output_norm(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


def zero_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
