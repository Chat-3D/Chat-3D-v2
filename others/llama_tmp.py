import json
import torch
import sys
sys.path.append(".")
from models.modeling_llama_new import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
from collections import defaultdict

from tqdm import tqdm


llama_model_path = "model/vicuna-7b-v0"

print("Loading LLaMA")
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
model = LlamaForCausalLM.from_pretrained(
    llama_model_path,
    torch_dtype=torch.float16
)
# model = model.to("cuda")
# print(torch.cuda.memory_allocated(device="cuda:0")/1e9)
# exit()
print("is training:", model.training)
# for p in model.parameters():
#     p.requires_grad = False
print("Loading LLaMA Done")

llama_tokenizer.add_tokens(["<obj01>", "<obj02>"], special_tokens=False)

model.resize_token_embeddings(len(llama_tokenizer))


def get_text_len(text):
    return llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]


def get_ids(text):
    return llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def get_emb(text, is_eval=False):
    input_ids = llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    if is_eval:
        model.eval()
    else:
        model.train()
    return model.model.embed_tokens(input_ids)


breakpoint()
