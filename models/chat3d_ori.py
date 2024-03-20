import random
import logging
from abc import ABC

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
from models.transformer_vanilla import TransformerEncoder, CMT
from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine, PositionalEmbedding

from transformers import StoppingCriteria, StoppingCriteriaList

import contextlib

logger = logging.getLogger(__name__)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


def init_weights(std=0.02):
    def _init_weights(module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    return _init_weights


class CustomGradLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, coefficient=1.0):
        ctx.coefficient = coefficient
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output * ctx.coefficient
        return grad_input, None


class Chat3D(nn.Module):
    """
    VideoChat model.
    """
    def __init__(self, config):
        super().__init__()
        llama_model_path = config.get("llama_model_path")
        low_resource = config.get("low_resource", False)
        # prompt
        self.prompt_template = config.get("prompt_template", "")
        self.max_txt_len = config.get("max_txt_len", 32)
        self.end_sym = config.get("end_sym", '\n')
        self.system_path = config.get("system_path", "")
        self.begin_signal = "###"
        self.role = ("Human", "Assistant")
        self.pc_start_token, self.pc_end_token = "<Target>", "</Target>"
        self.scene_start_token, self.scene_end_token = "<Scene>", "</Scene>"
        self.add_scene_token = config.get("add_scene_token", True)
        self.debug = config.get("debug", False)
        self.obj_norm_scale = config.get("obj_norm_scale", 1)
        self.scene_norm_scale = config.get("scene_norm_scale", 1)
        self.grad_scale = config.get("grad_scale", 1)

        mlp_dropout = config.get("mlp_dropout", 0.5)
        self.stage = config.get("stage", 1)

        self.low_resource = low_resource

        self.input_dim = config.get("input_dim", 512)
        self.attr_dim = config.get("attr_dim", 512)
        self.inter_dim = self.input_dim + self.attr_dim * 2

        if not self.debug:
            logger.info('Loading LLAMA')
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            if self.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map="auto"
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.float16,
                )
            logger.info("freeze LLAMA")
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            # if self.stage != 1:
            #     for layer_ind in range(30, 32):
            #         for param in self.llama_model.model.layers[layer_ind].parameters():
            #             param.requires_grad = True
            #             param.data = param.data.float()
            self.llama_dim = self.llama_model.config.hidden_size
            logger.info('Loading LLAMA Done')
        else:
            self.llama_model = None
            self.llama_dim = 4096

        # self.object_input_proj = nn.Sequential(
        #     nn.Linear(self.input_dim, self.input_dim),
        #     # nn.ReLU(),
        #     # nn.LayerNorm(self.input_dim),
        # )
        self.coord_proj = nn.Sequential(
            nn.Linear(3, self.attr_dim),
            # nn.ReLU(),
            # nn.LayerNorm(self.attr_dim),
            # nn.Dropout(mlp_dropout)
        )
        self.color_proj = nn.Sequential(
            nn.Linear(3, self.attr_dim),
            # nn.ReLU(),
            # nn.LayerNorm(self.attr_dim),
            # nn.Dropout(mlp_dropout)
        )
        # self.color_dropout = nn.Dropout(mlp_dropout)
        # self.pos_proj = nn.Sequential(
        #     nn.Linear(6, self.inter_dim),
        #     nn.LayerNorm(self.inter_dim)
        # )
        # self.pos_embedding = PositionalEmbedding(dim=self.llama_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, self.llama_dim)
        )
        self.object_proj = nn.Sequential(
            nn.Linear(self.inter_dim, self.llama_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.LayerNorm(self.llama_dim),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        self.scene_proj = nn.Sequential(
            nn.Linear(self.llama_dim, self.llama_dim),
        )
        self.encoder_num_layers = int(config.get("encoder_num_layers", 1))
        self.relation_module = CMT(hidden_size=self.llama_dim, num_layers=self.encoder_num_layers)
        # self.cls_head = nn.Sequential(
        #     nn.Linear(self.llama_dim, 40)
        # )

        if self.stage == 1:
            for p in self.relation_module.parameters():
                p.requires_grad = False
            for p in self.scene_proj.parameters():
                p.requires_grad = False
            for p in self.pos_proj.parameters():
                p.requires_grad = False
            # for p in self.pos_embedding.parameters():
            #     p.requires_grad = False
        # for p in self.relation_module.parameters():
        #     p.requires_grad = False
        # else:
        #     for p in self.size_color_proj.parameters():
        #         p.requires_grad = False
        #     for p in self.scene_proj.parameters():
        #         p.requires_grad = False
        # else:
        #     for p in self.size_color_proj.parameters():
        #         p.requires_grad = False
        #     for p in self.scene_proj.parameters():
        #         p.requires_grad = False

        with open(self.system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])

        if not self.debug:
            self.object_norm = torch.norm(self.get_text_emb("object"), p=2)
            self.relation_norm = torch.norm(self.get_text_emb("relation"), p=2)
            self.position_norm = torch.norm(self.get_text_emb("position"), p=2)
            if self.stage != 1:
                self.object_list_embed, self.object_list_ind = self.prepare_object_list()
                self.p_0_embed, self.p_1_embed = self.prepare_system_embed()

    # def process_prompt(self, prompt_path, prompt_template):
        # with open(prompt_path, 'r') as f:
        #     prompt_candidates = f.read().splitlines()
        # with open(self.system_path, "r") as f:
        #     system = "\n".join([x.strip() for x in f.readlines()])
        # prompt_list = [system + " " + prompt_template.format(p) for p in prompt_candidates]
        # logger.info(f'Load {len(prompt_list)} training prompts')
        # logger.info(f'Prompt: {prompt_list}')
        # return prompt_list

    # def prompt_wrap(self, scene_embed, scene_mask, prompts, is_eval=False):
    #     batch_size = scene_embed.shape[0]
    #     for i, prompt in enumerate(prompts):
    #         p_0, p_1 = prompt.split('<REPLACE>')
    #         p_0_tokens = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=is_eval).to(scene_embed.device)
    #         p_1_tokens = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False).to(scene_embed.device)
    #         # p_2_tokens = self.llama_tokenizer(p_2, return_tensors="pt", add_special_tokens=False).to(pc_embed.device)
    #         p_0_embeds = self.llama_model.model.embed_tokens(p_0_tokens.input_ids).expand(batch_size, -1, -1)
    #         p_1_embeds = self.llama_model.model.embed_tokens(p_1_tokens.input_ids).expand(batch_size, -1, -1)
    #         p_2_embeds = self.llama_model.model.embed_tokens(p_2_tokens.input_ids).expand(batch_size, -1, -1)
    #         wrapped_embeds = torch.cat([p_0_embeds, pc_embed, p_1_embeds, scene_embed, p_2_embeds], dim=1)
    #         wrapped_atts = scene_attn[:, :1].expand(-1, wrapped_embeds.shape[1])
    #     return wrapped_embeds, wrapped_atts

    # def get_object_list_embed(self, scene_embed, scene_mask):
    #     # scene_embed: (obj_num, dim)
    #     embed_list = []
    #     for i in range(scene_embed.shape[0]):
    #         if scene_mask[i] == 0:
    #             break
    #         text = ""
    #         if i > 0:
    #             text += ", "
    #         text += f"obj{i:02}: "
    #         text_embeds = self.get_text_emb(text, scene_embed.device).detach()
    #         embed_list.extend([text_embeds.squeeze(0), scene_embed[i:i+1]])
    #     return torch.cat(embed_list, dim=0)

    def prepare_object_list(self, max_obj_num=150):
        tmp_id = 0
        embed_list = []
        obj_index_list = []
        for i in range(max_obj_num):
            text = "" if i == 0 else "; "
            text += f"obj{i:02} "
            text_embeds = self.get_text_emb(text).squeeze(0)
            tmp_id += text_embeds.shape[0]
            obj_index_list.append(tmp_id)
            if self.add_scene_token:
                embed_list.extend([text_embeds, torch.zeros((2, text_embeds.shape[-1]))])
                tmp_id += 2
            else:
                embed_list.extend([text_embeds, torch.zeros((1, text_embeds.shape[-1]))])
                tmp_id += 1
        return torch.cat(embed_list, dim=0), obj_index_list

    def prepare_system_embed(self):
        prompt = self.system + " "
        p_0, p_1 = prompt.split("<REPLACE>")
        p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=False)
        p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False)
        p_0_embed = self.llama_model.model.embed_tokens(p_0_token.input_ids).squeeze(0)
        p_1_embed = self.llama_model.model.embed_tokens(p_1_token.input_ids).squeeze(0)
        return p_0_embed, p_1_embed

    def get_text_emb(self, text, device="cpu"):
        text_tokens = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        return self.llama_model.model.embed_tokens(text_tokens.input_ids)

    def encode_object_feat(self, feat, locs, colors):
        # feat = self.object_input_proj(feat)
        size_emb = self.coord_proj(locs[:, :, 3:6])
        gmm_weights = colors[..., :1]
        gmm_means = colors[..., 1:]
        gmm_colors = torch.sum(gmm_weights * gmm_means, dim=2)
        # color_emb = self.color_dropout(torch.sum(self.color_proj(gmm_means) * gmm_weights, dim=2))
        color_emb = self.color_proj(gmm_colors)
        feat = torch.cat([feat, size_emb, color_emb], dim=-1)
        # feat = torch.cat([feat, size_emb], dim=-1)
        # feat = self.scene_proj(feat)
        return feat

    @staticmethod
    def get_dist_attention(pos, dist_exp=1):
        # pos (bs, obj_num, 3)
        dist = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.sum(dist.abs()**dist_exp, dim=-1)
        dist_attn = torch.nn.functional.softmax(-dist, dim=-1)
        return dist_attn

    def insert_object_embed(self, embed_1, embed_2, scene_mask, detach_mask=None):
        if detach_mask is not None:
            embed_1_detached = CustomGradLayer.apply(embed_1[detach_mask], self.grad_scale)
            embed_1[detach_mask] = embed_1_detached
            if embed_2 is not None:
                embed_2_detached = CustomGradLayer.apply(embed_2[detach_mask], self.grad_scale)
                embed_2[detach_mask] = embed_2_detached
        obj_num = int(scene_mask.sum())
        mx_ind = self.object_list_ind[obj_num - 1] + (2 if self.add_scene_token else 1)
        object_list_embed = self.object_list_embed[:mx_ind, :].to(embed_1.device)
        object_list_ind = torch.tensor(self.object_list_ind[:obj_num], dtype=torch.long)\
            .to(embed_1.device)
        object_list_embed[object_list_ind] = embed_1[scene_mask.bool()].to(object_list_embed.dtype)
        if self.add_scene_token:
            object_list_embed[object_list_ind+1] = embed_2[scene_mask.bool()].to(object_list_embed.dtype)
        return object_list_embed

    def forward_stage1(self, scene_feat, scene_locs, scene_colors, target_captions, is_eval=False, **kwargs):
        object_embed = self.encode_object_feat(scene_feat, scene_locs, scene_colors)
        proj_object_embed = self.object_proj(object_embed)
        proj_object_embed = proj_object_embed.squeeze(1)
        # cls_output = self.cls_head(proj_object_embed)
        # cls_loss = F.cross_entropy(cls_output, target_clses)
        # cls_acc = (cls_output.max(dim=-1)[1] == target_clses).float().mean()
        norm_object_embed = torch.nn.functional.normalize(proj_object_embed, dim=-1) * self.obj_norm_scale
        target_embeds = []
        for target_caption in target_captions:
            target_tokens = self.llama_tokenizer(
                target_caption,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(norm_object_embed.device)
            token_mask = target_tokens["attention_mask"].unsqueeze(-1)
            target_embed = self.llama_model.model.embed_tokens(target_tokens.input_ids)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            target_embed = (target_embed * token_mask).sum(1) / token_mask.sum(1)
            target_embed = target_embed.mean(dim=0)
            target_embeds.append(target_embed)
        target_embeds = torch.stack(target_embeds, dim=0).to(norm_object_embed.device)
        cosine_loss = F.cosine_embedding_loss(norm_object_embed, target_embeds.detach(), torch.tensor([1]).to(norm_object_embed.device))
        l2_loss = F.mse_loss(proj_object_embed, target_embeds.detach())
        # print(torch.norm(pc_embed[:1], p=2), torch.norm(target_embeds[:1], p=2))
        loss = cosine_loss
        return dict(
            loss=loss,
            cosine_loss=cosine_loss,
            # cls_loss=cls_loss,
            l2_loss=l2_loss,
            # cls_acc=cls_acc.detach().cpu(),
            cosine_score=1. - cosine_loss.detach().cpu(),
            obj_norm=proj_object_embed.norm(dim=-1).mean().detach().cpu(),
            target_norm=target_embeds.norm(dim=-1).mean().detach().cpu(),
            l2_dis=l2_loss.detach().cpu()
        )

    def forward_stage2(self, scene_feat, scene_locs, scene_colors, scene_mask, detach_mask, obj_ids, questions, answers, is_eval=False, **kwargs):
        object_embed = self.encode_object_feat(scene_feat, scene_locs, scene_colors)
        device = object_embed.device
        batch_size = object_embed.shape[0]
        proj_object_embed = self.object_proj(object_embed)
        norm_object_embed = torch.nn.functional.normalize(proj_object_embed, dim=-1) * self.obj_norm_scale
        # norm_object_embed = proj_object_embed
        proj_scene_embed = None
        if self.add_scene_token:  # remember to change the evaluate !!!!!!!!!!!!!!!!!!!!!!(&*&^^#@$%##$%&(*^&%^$%@
            pos_embed = self.pos_proj(scene_locs[:, :, :3])
            scene_embed = proj_object_embed.detach() + 0.1 * pos_embed

            # scene_embed = scene_embed.mean(dim=1, keepdim=True).repeat(1, scene_embed.shape[1], 1)
            # proj_scene_embed = scene_embed - proj_object_embed

            scene_embed = self.relation_module(scene_embed, scene_locs, scene_mask.bool())
            proj_scene_embed = scene_embed - proj_object_embed.detach()

            norm_scene_embed = torch.nn.functional.normalize(proj_scene_embed, dim=-1) * self.scene_norm_scale
            # norm_scene_embed = proj_scene_embed
        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        for i, question in enumerate(questions):
            prompt = self.prompt_template.format(question)
            prompt_token = self.llama_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            prompt_embed = self.llama_model.model.embed_tokens(prompt_token.input_ids).detach().squeeze(0)
            # object_list_embed = self.get_object_list_embed(scene_embed[i], scene_mask[i])
            detach_mask = None
            object_list_embed = self.insert_object_embed(norm_object_embed[i], norm_scene_embed[i] if self.add_scene_token else None, scene_mask[i], detach_mask[i] if detach_mask is not None else None)
            # for j in range(obj_num):
            #     start_ind = self.object_list_ind[j]
            #     assert object_list_embed[start_ind].abs().sum() < 1e-6, (start_ind, object_list_embed[start_ind].sum())
            #     assert object_list_embed[start_ind+1].abs().sum() < 1e-6, (start_ind+1, object_list_embed[start_ind+1].sum())
            #     object_list_embed[start_ind:start_ind+1, :] = scene_embed[i][j]
            #     object_list_embed[start_ind+1:start_ind+2, :] = pos_embed[i][j]

            p_0_embed = self.p_0_embed.to(device)
            p_1_embed = self.p_1_embed.to(device)

            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=0)
            wrapped_attn = torch.ones(wrapped_embed.size()[:-1], dtype=torch.long).to(wrapped_embed.device)
            empty_target = (
                torch.ones([wrapped_attn.shape[0]+1], dtype=torch.long).to(device).fill_(-100)
            )

            answer = answers[i] + self.end_sym
            to_regress_token = self.llama_tokenizer(answer, return_tensors="pt").to(device)
            # breakpoint()
            answer_target = to_regress_token.input_ids.masked_fill(
                to_regress_token.input_ids == self.llama_tokenizer.pad_token_id, -100
            ).squeeze(0)
            to_regress_embed = self.llama_model.model.embed_tokens(to_regress_token.input_ids).squeeze(0)

            target = torch.cat([empty_target, answer_target], dim=0)
            bos = torch.ones([1], dtype=to_regress_token.input_ids.dtype, device=to_regress_token.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embed = self.llama_model.model.embed_tokens(bos)
            bos_attn = wrapped_attn[:1]
            input_embed = torch.cat([bos_embed, wrapped_embed, to_regress_embed], dim=0)
            attn = torch.cat([bos_attn, wrapped_attn, to_regress_token.attention_mask[0]], dim=0)
            input_embed_list.append(input_embed)
            attn_list.append(attn)
            target_list.append(target)
            max_seq_len = max(max_seq_len, target.shape[0])
            

        dim = norm_object_embed.shape[2]
        max_seq_len = min(256, max_seq_len)

        input_embeds = torch.zeros([batch_size, max_seq_len, dim], dtype=input_embed_list[0].dtype).to(device)
        attention_mask = torch.zeros([batch_size, max_seq_len], dtype=attn_list[0].dtype).to(device)
        targets = torch.zeros([batch_size, max_seq_len], dtype=target_list[0].dtype).to(device).fill_(-100)
        for i in range(len(input_embed_list)):
            input_embed = input_embed_list[i]
            attn = attn_list[i]
            target = target_list[i]
            input_embeds[i, :min(input_embed.shape[0], max_seq_len), :] = input_embed[:min(input_embed.shape[0], max_seq_len)]
            attention_mask[i, :min(attn.shape[0], max_seq_len)] = attn[:min(input_embed.shape[0], max_seq_len)]
            targets[i, :min(target.shape[0], max_seq_len)] = target[:min(input_embed.shape[0], max_seq_len)]
        
        # if max_seq_len > 256:
        #     input_embeds = input_embeds[:, :256]
        #     attention_mask = attention_mask[:, :256]
        #     targets = targets[:, :256]

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        return dict(
            loss=outputs.loss,
            obj_norm=proj_object_embed.norm(dim=-1).mean().detach().cpu(),
            scene_norm=proj_scene_embed.norm(dim=-1).mean().detach().cpu() if proj_scene_embed is not None else 0.,
            max_seq_len=max_seq_len
        )

    def forward_stage3(self, scene_feat, scene_attr, scene_mask, target_id, conversations, is_eval=False, **kwargs):
        batch_size, obj_num, _ = scene_feat.shape
        scene_feat = self.encode_and_project(scene_feat, scene_attr)
        pc_embed = torch.gather(scene_feat, 1, target_id.unsqueeze(1).unsqueeze(2).expand(-1, -1, scene_feat.shape[-1]))
        if self.encoder_num_layers > 0:
            scene_feat = self.relation_module(scene_feat, mask=(~scene_mask.bool()).unsqueeze(1).expand(-1, obj_num, -1).unsqueeze(1))

        scene_embed = scene_feat * scene_mask.unsqueeze(-1)
        # scene_attn = torch.ones(scene_embed.size()[:-1], dtype=torch.long).to(scene_embed.device)
        max_len = 0
        input_embed_list = []
        p_0_len_list, p_1_len_list = [], []
        target_list = []
        for idx, prompt in enumerate(conversations):
            tmp_scene_embed = scene_embed[idx:idx+1]
            tmp_pc_embed = pc_embed[idx:idx+1]
            p_0, p_ = prompt.split("<TargetHere>")
            p_1, p_2 = p_.split("<SceneHere>")
            p_1 = self.pc_end_token + p_1
            p_0_tokens = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=is_eval).to(tmp_pc_embed.device)
            p_1_tokens = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False).to(tmp_pc_embed.device)
            p_2_tokens = self.llama_tokenizer(p_2, return_tensors="pt", add_special_tokens=False).to(tmp_pc_embed.device)
            p_0_embeds = self.llama_model.model.embed_tokens(p_0_tokens.input_ids)
            p_1_embeds = self.llama_model.model.embed_tokens(p_1_tokens.input_ids)
            p_2_embeds = self.llama_model.model.embed_tokens(p_2_tokens.input_ids)
            input_embeds = torch.cat([p_0_embeds, tmp_pc_embed, p_1_embeds, tmp_scene_embed, p_2_embeds], dim=1)

            sep1 = self.begin_signal + self.role[0] + ":"
            sep2 = self.begin_signal + self.role[1] + ":"
            raw_text = p_2.split(sep2)
            for _idx in range(1, len(raw_text)):
                raw_text[_idx] = sep2 + raw_text[_idx]
            answer_targets = p_2_tokens.input_ids.clone()
            system = raw_text[0].split(sep1)[0]
            system_len = self._get_text_len(system.rstrip())
            sep_len = self._get_text_len(sep1.rstrip())
            cur_len = self._get_text_len(raw_text[0].rstrip())
            answer_targets[:, :system_len] = -100
            answer_targets[:, (system_len+sep_len):cur_len] = -100
            for text in raw_text[1:-1]:
                total_len = self._get_text_len(text.rstrip())
                ans_len = self._get_text_len((text.split(sep1)[0]+sep1).rstrip())
                answer_targets[:, (cur_len+ans_len):(cur_len+total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1].rstrip())
            if cur_len != answer_targets.shape[1]:
                print(f"The final length is not equal to the original prompt: {prompt}")
            assert cur_len == answer_targets.shape[1], (cur_len, answer_targets.shape[1])

            max_len = max(max_len, input_embeds.shape[1])
            input_embed_list.append(input_embeds)
            p_0_len_list.append(p_0_tokens.input_ids.shape[1])
            p_1_len_list.append(p_1_tokens.input_ids.shape[1])
            target_list.append(answer_targets)

        txt_len = min(max_len + 1, self.max_txt_len + obj_num + 1)
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(pc_embed.device) * self.llama_tokenizer.pad_token_id
        inputs_embeds = self.llama_model.model.embed_tokens(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(pc_embed.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(pc_embed.device).fill_(-100)
        inputs_embeds[:, :1] = self.llama_tokenizer.bos_token_id
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            inputs_embeds[idx, 1:(input_len+1)] = input_embed_list[idx][:, :input_len]
            attention_mask[idx, :(input_len+1)] = 1
            p_0_len = p_0_len_list[idx]
            p_1_len = p_1_len_list[idx]
            targets[idx, (p_0_len+p_1_len+obj_num+2):(input_len+1)] = target_list[idx][0, :(input_len-p_0_len-p_1_len-obj_num-1)]

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets
        )

        return dict(
            loss=outputs.loss
        )

    def evaluate(self, scene_feat, scene_locs, scene_colors, scene_mask, custom_prompt, is_eval=True, **kwargs):
        object_embed = self.encode_object_feat(scene_feat, scene_locs, scene_colors)
        device = object_embed.device
        batch_size, obj_num = object_embed.shape[:2]
        proj_object_embed = self.object_proj(object_embed)
        norm_object_embed = torch.nn.functional.normalize(proj_object_embed, dim=-1) * self.obj_norm_scale
        # norm_object_embed = proj_object_embed
        if self.add_scene_token:
            pos_embed = self.pos_proj(scene_locs[:, :, :3])
            scene_embed = proj_object_embed + 0.1 * pos_embed

            # scene_embed = scene_embed.mean(dim=1, keepdim=True).repeat(1, scene_embed.shape[1], 1)
            # proj_scene_embed = scene_embed - proj_object_embed

            scene_embed = self.relation_module(scene_embed, scene_locs, scene_mask.bool())
            proj_scene_embed = scene_embed - proj_object_embed

            norm_scene_embed = torch.nn.functional.normalize(proj_scene_embed, dim=-1) * self.scene_norm_scale
            # norm_scene_embed = proj_scene_embed

        output_texts = []
        for i in range(batch_size):
            # tmp_scene_embed, _ = self.prompt_wrap(pc_embed[idx:idx+1], scene_embed[idx:idx+1], scene_attn[idx:idx+1], custom_prompt[idx], is_eval)
            p_0, p_1 = custom_prompt[i].split("<REPLACE>")
            p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=is_eval).to(device)
            p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False).to(device)
            p_0_embed = self.llama_model.model.embed_tokens(p_0_token.input_ids)
            p_1_embed = self.llama_model.model.embed_tokens(p_1_token.input_ids)

            object_list_embed = self.insert_object_embed(norm_object_embed[i], norm_scene_embed[i] if self.add_scene_token else None, scene_mask[i])

            # for j in range(obj_num):
            #     start_ind = self.object_list_ind[j]
            #     object_list_embed[start_ind:start_ind + 1, :] = scene_embed[i][j]
            #     object_list_embed[start_ind + 1:start_ind + 2, :] = pos_embed[i][j]
            object_list_embed = object_list_embed.unsqueeze(0)
            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed], dim=1)
            stop_words_ids = [torch.tensor([835]).to(wrapped_embed.device),
                              torch.tensor([2277, 29937]).to(wrapped_embed.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
            with self.maybe_autocast():
                outputs = self.llama_model.generate(
                    inputs_embeds=wrapped_embed,
                    max_new_tokens=min(self.max_txt_len * 2, 512),
                    stopping_criteria=stopping_criteria,
                    num_beams=1,
                    do_sample=True,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    length_penalty=1,
                    temperature=1.0,
                )
            output_token = outputs[0]
            if output_token[0] == 0:  # the model might output an unknown token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split(self.end_sym)[0]
            output_texts.append(output_text)

        return output_texts

    def forward(self, **kwargs):
        if "target_captions" in kwargs:
            return self.forward_stage1(**kwargs)
        if "answers" in kwargs:
            return self.forward_stage2(**kwargs)
        if "conversations" in kwargs:
            return self.forward_stage3(**kwargs)
        if "custom_prompt" in kwargs:
            return self.evaluate(**kwargs)
        return None

    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device


# class CustomBackwardFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, scaling_factor):
#         # Save the scaling factor for later use in backward pass
#         ctx.scaling_factor = scaling_factor
#         return input
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         # Scale the gradients by the previously saved scaling factor
#         return grad_output * ctx.scaling_factor, None
#
#
# def scale_gradients(grad):
#     return grad * 10.