import random
import logging
from abc import ABC

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from .modeling_llama_new import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
from models.transformer_vanilla import TransformerEncoder, CMT
from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine, PositionalEmbedding
from peft import LoraConfig, get_peft_model
from transformers.tokenization_utils_base import AddedToken
# from models.load_llama import init_llama_model
from torch.nn.utils.rnn import pad_sequence

from transformers import StoppingCriteria, StoppingCriteriaList
from IPython import embed
import contextlib
import math

logger = logging.getLogger(__name__)


def nclamp(input, min, max):
    return input.clamp(min=min, max=max).detach() + input - input.detach()


def print_grad_status(model):
    """Call this function after losses.backward()
    and it will find out all variables without grad, which
    means that the varaible is not in the graph.
    """
    for name, p in model.named_parameters():
        print('{:80s}{:20s}{:20s}{}'.format(name,
            '(Trainable)' if p.requires_grad else '(Fixed)',
            '(Has grad):' if p.grad is not None else '(No grad backward):',
            list(p.shape)))


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
        llama_model_path = config.model.llama_model_path
        self.low_resource = config.model.low_resource
        self.max_txt_len = config.model.max_txt_len
        self.end_sym = config.model.end_sym
        self.system_path = config.model.system_path
        self.instruction_path = config.model.instruction_path
        self.role = config.model.role
        self.no_obj = config.model.no_obj
        self.add_scene_token = config.model.add_scene_token
        self.add_img_token = config.model.add_img_token
        self.obj_norm_scale = config.model.obj_norm_scale
        self.scene_norm_scale = config.model.scene_norm_scale
        self.grad_scale = config.model.grad_scale
        self.train_emb = config.model.train_emb
        self.train_img_proj = config.model.train_img_proj

        mlp_dropout = config.model.mlp_dropout
        self.stage = config.model.stage

        self.input_dim = config.model.input_dim
        self.img_input_dim = config.model.img_input_dim
        self.attr_dim = config.model.attr_dim
        self.scene_dim = config.model.scene_dim
        self.inter_dim = self.input_dim + self.attr_dim * 2

        # self.pc_start_token, self.pc_end_token = "<Target>", "</Target>"
        # self.scene_start_token, self.scene_end_token = "<Scene>", "</Scene>"

        # self.llama_tokenizer, self.llama_model = init_llama_model(config)
        self.debug = config.debug
        if not self.debug:
            logger.info('Loading LLAMA')
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False, legacy=False)
            # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            if self.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map="auto",
                    attn_implementation="flash_attention_2"
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                )
            # print(torch.cuda.memory_allocated(device="cuda:0")/1e9)
            # self.llama_model = self.llama_model.to("cuda")
            # print(torch.cuda.memory_allocated(device="cuda:0")/1e9)
            # breakpoint()
            logger.info("freeze LLAMA")
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            self.llama_model.lm_head.weight.requires_grad = True
            self.llama_model.lm_head.weight.data = self.llama_model.lm_head.weight.data.float()
            self.llama_model.model.embed_tokens.weight.requires_grad = True
            self.llama_model.model.embed_tokens.weight.data = self.llama_model.model.embed_tokens.weight.data.float()

            if config.model.use_lora:
                def find_linear_layers(model, lora_target_modules):
                    cls = torch.nn.Linear
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        if (
                            isinstance(module, cls)
                            and all(
                                [
                                    x not in name
                                    for x in [
                                        "instance2embed",
                                        "hidden_state2query"
                                    ]
                                ]
                            )
                            and any([x in name for x in lora_target_modules])
                        ):
                            lora_module_names.add(name)
                            # print(f"add lora to {name}")
                    return sorted(list(lora_module_names))
            
                lora_target_modules = find_linear_layers(self.llama_model, config.lora.lora_target_modules)

                lora_config = LoraConfig(
                    r=config.lora.lora_r,
                    lora_alpha=config.lora.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=config.lora.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llama_model = get_peft_model(self.llama_model, lora_config)
                self.llama_model.print_trainable_parameters()
            
            self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

            objid_tokens = []
            for i in range(200):
                objid_tokens.append(f"<OBJ{i:03}>")
            self.objid_start_idx = self.ori_vocab_size = len(self.llama_tokenizer)
            self.llama_tokenizer.add_tokens(objid_tokens, special_tokens=True)
            self.objid_end_idx = len(self.llama_tokenizer)
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

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
        # self.coord_proj = nn.Sequential(
        #     nn.Linear(3, self.attr_dim),
        #     # nn.ReLU(),
        #     # nn.LayerNorm(self.attr_dim),
        #     # nn.Dropout(mlp_dropout)
        # )
        # self.color_proj = nn.Sequential(
        #     nn.Linear(3, self.attr_dim),
        #     # nn.ReLU(),
        #     # nn.LayerNorm(self.attr_dim),
        #     # nn.Dropout(mlp_dropout)
        # )
        # self.color_dropout = nn.Dropout(mlp_dropout)
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=self.scene_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(self.scene_dim, self.scene_dim)
        )
        self.object_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        self.object_img_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        if not self.train_img_proj:
            for p in self.object_img_proj.parameters():
                p.requires_grad = False
        # self.object_layer_norm = nn.LayerNorm(self.input_dim)
        # self.scene_proj = nn.Sequential(
        #     nn.Linear(self.llama_dim, self.llama_dim),
        # )
        # self.encoder_num_layers = config.model.encoder_num_layers
        # self.relation_module = CMT(hidden_size=self.llama_dim, num_layers=self.encoder_num_layers)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.scene_dim, nhead=8, dim_feedforward=2048, norm_first=True, batch_first=True)
        self.relation_module = nn.TransformerEncoder(self.encoder_layer, num_layers=config.model.encoder_num_layers)
        self.scene_init_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.scene_dim)
        )
        self.scene_proj = nn.Sequential(
            nn.Linear(self.scene_dim, self.llama_dim),
            # nn.GELU(),
            # nn.Linear(self.llama_dim, self.llama_dim)
        )
        
        if not self.add_scene_token:
            for p in self.relation_module.parameters():
                p.requires_grad = False
            for p in self.scene_init_proj.parameters():
                p.requires_grad = False
            for p in self.scene_proj.parameters():
                p.requires_grad = False

        if self.stage == 1:
            for p in self.relation_module.parameters():
                p.requires_grad = False
            for p in self.pos_proj.parameters():
                p.requires_grad = False
                

        with open(self.system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        with open(self.instruction_path, "r") as f:
            self.instruction = "\n".join([x.strip() for x in f.readlines()])

        if not self.debug:
            self.object_norm = torch.norm(self.get_text_emb("object"), p=2)
            self.relation_norm = torch.norm(self.get_text_emb("relation"), p=2)
            self.position_norm = torch.norm(self.get_text_emb("position"), p=2)
            if self.stage != 1:
                # self.object_list_embed, self.object_list_ind = self.prepare_object_list()
                self.p_0_embed, self.p_1_embed = self.prepare_fixed_embed()
        self.last_embed = None
        
        # print_grad_status(self)

    def prepare_fixed_embed(self):
        prompt = self.system + " " + self.instruction + ' ' + self.role[0] + ": " 
        p_0, p_1 = prompt.split("<REPLACE>")
        p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=True)
        p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False)
        p_0_embed = self.llama_model.model.embed_tokens(p_0_token.input_ids).squeeze(0).detach()
        p_1_embed = self.llama_model.model.embed_tokens(p_1_token.input_ids).squeeze(0).detach()
        return p_0_embed, p_1_embed

    def get_text_emb(self, text, device="cpu"):
        text_tokens = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)
        if self.train_emb:
            indices = text_tokens.input_ids >= self.ori_vocab_size
            indices = (indices * 1).unsqueeze(-1)
            embeds = (1 - indices) * embeds.detach() + indices * embeds
        else:
            embeds = embeds.detach()
        return embeds

    def encode_object_feat(self, feat, img_feat, locs):
        # size_emb = self.coord_proj(locs[:, :, 3:6])
        # gmm_weights = colors[..., :1]
        # gmm_means = colors[..., 1:]
        # gmm_colors = torch.sum(gmm_weights * gmm_means, dim=2)
        # color_emb = self.color_proj(gmm_colors)
        # feat = torch.cat([feat, size_emb, color_emb], dim=-1)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
        return feat, img_feat
    
    @staticmethod
    def get_dist_attention(pos, dist_exp=1):
        # pos (bs, obj_num, 3)
        dist = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.sum(dist.abs()**dist_exp, dim=-1)
        dist_attn = torch.nn.functional.softmax(-dist, dim=-1)
        return dist_attn

    def get_object_list_embed(self, embed_obj, embed_img, embed_scene, scene_mask):
        valid_ids = torch.where(scene_mask == 1)[0].tolist()
        if len(valid_ids) == 1:
            object_list_embed = []
            objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]
            object_list_embed.append(objid_embeds[random.randint(0, 199)])
            object_list_embed.append(embed_obj[0])
            # if random.randint(0, 1) == 0:
            #     object_list_embed.append(embed_scene[0])
            object_list_embed = torch.stack(object_list_embed, dim=0)
        
        random.shuffle(valid_ids)
        objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]  # 200 * 4096
        if not self.train_emb:
            objid_embeds = objid_embeds.detach()
        selected_objid_embeds = objid_embeds[valid_ids]
        # if self.no_obj:
        #     object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
        #     object_list_embed[0::3, :] = selected_objid_embeds
        #     object_list_embed[1::3, :] = embed_img[valid_ids]
        #     object_list_embed[2::3, :] = embed_scene[valid_ids]
        #     return object_list_embed
        if embed_img is None and embed_scene is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            object_list_embed[1::2, :] = embed_obj[valid_ids]
        if embed_img is None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[valid_ids]
            object_list_embed[2::3, :] = embed_scene[valid_ids]
        if embed_img is not None and embed_scene is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[valid_ids]
            object_list_embed[2::3, :] = embed_img[valid_ids]
        if embed_img is not None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 4, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::4, :] = selected_objid_embeds
            object_list_embed[1::4, :] = embed_obj[valid_ids]
            object_list_embed[2::4, :] = embed_img[valid_ids]
            object_list_embed[3::4, :] = embed_scene[valid_ids]
        return object_list_embed

    def forward_stage1(self, scene_feat, scene_locs, scene_colors, target_captions, is_eval=False, **kwargs):
        object_embed = self.encode_object_feat(scene_feat, scene_locs, scene_colors)
        proj_object_embed = self.object_proj(object_embed)
        proj_object_embed = proj_object_embed.squeeze(1)
        # cls_output = self.cls_head(proj_object_embed)
        # cls_loss = F.cross_entropy(cls_output, target_clses)
        # cls_acc = (cls_output.max(dim=-1)[1] == target_clses).float().mean()
        # norm_object_embed = torch.nn.functional.normalize(proj_object_embed, dim=-1) * self.obj_norm_scale
        norm_object_embed = proj_object_embed
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
    
    def get_min_max_coord(self, xyz, scene_mask):
        scene_mask = scene_mask.unsqueeze(-1).expand_as(xyz)
        masked_xyz_min = torch.where(scene_mask > 0, xyz, torch.full_like(xyz, float('inf')))
        masked_xyz_max = torch.where(scene_mask > 0, xyz, torch.full_like(xyz, float('-inf')))
        mins = masked_xyz_min.min(dim=1)[0]
        maxs = masked_xyz_max.max(dim=1)[0]
        return mins, maxs

    def forward_stage2(self, scene_feat, scene_img_feat, scene_locs, scene_mask, obj_ids, questions, answers, is_eval=False, **kwargs):
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        batch_size = object_embed.shape[0]
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = self.object_img_proj(object_img_embed)

        proj_scene_embed = None
        if self.add_scene_token:  # remember to change the evaluate 
            # if self.add_img_token:
            #     object_embed = object_embed + object_img_embed
            obj_embed = self.scene_init_proj(object_embed)
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed)
            scene_embed = obj_embed + pos_embed
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~(scene_mask.bool()))
            proj_scene_embed = self.scene_proj(scene_embed)
        
        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        p_0_embed = self.p_0_embed.to(device)
        p_1_embed = self.p_1_embed.to(device)

        for i, question in enumerate(questions):
            prompt = f" {question} {self.role[1]}: "
            prompt_embed = self.get_text_emb(prompt, device=device).squeeze(0)
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i], 
                proj_object_img_embed[i] if self.add_img_token else None, 
                proj_scene_embed[i] if self.add_scene_token else None, 
                scene_mask[i]
            )
            # object_list_embed = nclamp(object_list_embed, min=-0.05, max=0.05)
            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=0)
            wrapped_attn = torch.ones(wrapped_embed.size()[:-1], dtype=torch.long).to(wrapped_embed.device)
            empty_target = (
                torch.ones(wrapped_attn.shape[0], dtype=torch.long).to(device).fill_(-100)
            )

            answer = answers[i] + self.end_sym
            to_regress_token = self.llama_tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(device)
            # breakpoint()
            answer_target = to_regress_token.input_ids.masked_fill(
                to_regress_token.input_ids == self.llama_tokenizer.pad_token_id, -100
            ).squeeze(0)
            # to_regress_embed = self.llama_model.model.embed_tokens(to_regress_token.input_ids).squeeze(0).detach()
            to_regress_embed = self.get_text_emb(answer, device=device).squeeze(0)

            target = torch.cat([empty_target, answer_target], dim=0)
            input_embed = torch.cat([wrapped_embed, to_regress_embed], dim=0)
            attn = torch.cat([wrapped_attn, to_regress_token.attention_mask[0]], dim=0)
            input_embed_list.append(input_embed)
            attn_list.append(attn)
            target_list.append(target)
            max_seq_len = max(max_seq_len, target.shape[0])
        
        max_seq_len = min(768, max_seq_len)

        def pad_and_trim(tensor_list, max_len, batch_first=True, padding_value=0):
            padded = pad_sequence(tensor_list, batch_first=batch_first, padding_value=padding_value)
            if padded.shape[1] > max_len:
                return padded[:, :max_len]
            return padded
        
        input_embeds = pad_and_trim(input_embed_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        attention_mask = pad_and_trim(attn_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        targets = pad_and_trim(target_list, max_seq_len, batch_first=True, padding_value=-100).to(device)


        
        # input_embeds = torch.zeros([batch_size, max_seq_len, dim], dtype=input_embed_list[0].dtype).to(device)
        # attention_mask = torch.zeros([batch_size, max_seq_len], dtype=attn_list[0].dtype).to(device)
        # targets = torch.zeros([batch_size, max_seq_len], dtype=target_list[0].dtype).to(device).fill_(-100)
        # for i in range(len(input_embed_list)):
        #     input_embed = input_embed_list[i]
        #     attn = attn_list[i]
        #     target = target_list[i]
        #     input_embeds[i, :min(input_embed.shape[0], max_seq_len), :] = input_embed[:min(input_embed.shape[0], max_seq_len)]
        #     attention_mask[i, :min(attn.shape[0], max_seq_len)] = attn[:min(input_embed.shape[0], max_seq_len)]
        #     targets[i, :min(target.shape[0], max_seq_len)] = target[:min(input_embed.shape[0], max_seq_len)]

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets
            )

        return dict(
            loss=outputs.loss,
            obj_norm=proj_object_embed.norm(dim=-1).mean().detach().cpu(),
            obj_img_norm=proj_object_img_embed.norm(dim=-1).mean().detach().cpu(),
            scene_norm=proj_scene_embed.norm(dim=-1).mean().detach().cpu() if proj_scene_embed is not None else 0.,
            max_seq_len=max_seq_len
        )

    def evaluate(self, scene_feat, scene_img_feat, scene_locs, scene_mask, custom_prompt, is_eval=True, **kwargs):
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        batch_size, obj_num = object_embed.shape[:2]
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = self.object_img_proj(object_img_embed)
        if self.add_scene_token:
            # if self.add_img_token:
            #     object_embed = object_embed + object_img_embed
            obj_embed = self.scene_init_proj(object_embed)
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed)
            scene_embed = obj_embed + pos_embed
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~(scene_mask.bool()))
            proj_scene_embed = self.scene_proj(scene_embed)

        output_texts = []
        p_0_embed = self.p_0_embed.to(device).unsqueeze(0)
        p_1_embed = self.p_1_embed.to(device).unsqueeze(0)
        for i in range(batch_size):
            tmp_prompt = f" {custom_prompt[i]} {self.role[1]}: "
            prompt_embed = self.get_text_emb(tmp_prompt, device=device)
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i], 
                proj_object_img_embed[i] if self.add_img_token else None, 
                proj_scene_embed[i] if self.add_scene_token else None, 
                scene_mask[i]
            )
            object_list_embed = object_list_embed.unsqueeze(0)
            # object_list_embed = nclamp(object_list_embed, min=-0.05, max=0.05)
            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=1)
            with self.maybe_autocast():
                outputs = self.llama_model.generate(
                    inputs_embeds=wrapped_embed,
                    max_new_tokens=self.max_txt_len,
                    # stopping_criteria=stopping_criteria,
                    num_beams=3,
                    # do_sample=True,
                    min_length=1,
                    # top_p=0.9,
                    repetition_penalty=1.0,
                    length_penalty=1,
                    temperature=1.0,
                )
            output_token = outputs[0]
            output_text = self.llama_tokenizer.decode(output_token)
            output_text = output_text.split(self.end_sym)[0]
            output_text = output_text.replace('  ', ' ').replace(' .', '.').strip()
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
        return self.llama_tokenizer(text, return_tensors="pt").input_ids.shape[1]

    def maybe_autocast(self, dtype=torch.bfloat16):
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
