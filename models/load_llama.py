import json

from transformers import LlamaTokenizer, AutoConfig

def init_llama_model(config):
    llama_model = None
    if config.pretrained_path:
        llama_config = AutoConfig.from_pretrained(config.pretrained_path)
        llama_config.first_init = False
        llama_config.save_path = config.
        logger.warning(f"load LLM config from {self.config['general']['save_dir']}" )
    else:
        print('*****************************************************************')
        print('*****************************************************************')
        print(f'Using **ACTUALLY** config: {self.config.general.llm_config}')
        print(f'Using **ACTUALLY** data config: {self.config.general.llm_data_config}')
        llama_config = AutoConfig.from_pretrained(self.config.general.llm_config)
        assert llama_config.vicuna_version == llama_config.vicuna_version, "conflict model"
        self.data_to_load = json.load(open(self.config.general.llm_data_config))
        print('*****************************************************************')
        print('*****************************************************************')
        llama_config.save_path = f"{self.config['general']['save_dir']}/{save_folder_name}"
        if self.global_rank == 0:
            llama_config.save_pretrained(f"{self.config['general']['save_dir']}")
            # logger.warning(
            #     f"current dir does not contain a config.json for LLM, thus the default config in trainer/trainer.py is used."
            # )
    if llama_config.enable_llm:
        assert self.config.model.num_queries == llama_config.num_of_queries,"llm should be trained and evaluated on same number of queries"
    assert llama_config.instance_query or llama_config.scene_aware_query," at least one point cloud feature should be used"
    os.makedirs(llama_config.save_path, exist_ok=True)
    os.makedirs(f"{llama_config.save_path}/m3drefer", exist_ok=True)

    if not llama_config.load_pretrain_weight:
        logger.warning(
            f"llm pretrain weight is not loaded: do you need to debug or resume from last_epoch.ckpt !?"
        )
    
    if llama_config.enable_llm:
        assert not self.config.general.use_dbscan
        # init tokenizer and add special tokens

        if llama_config.vicuna_version == "TinyLlama-1.1B-intermediate-step-1195k-token-2.5T" or llama_config.vicuna_version == "Tiny-Vicuna-1B":
            llama_config.vicuna_weight_path = [f"{llama_config.root_path}{llama_config.vicuna_version}/pytorch_model.bin"]
        elif llama_config.vicuna_version == "vicuna-7b-v1.5":
            llama_config.vicuna_weight_path = [f"{llama_config.root_path}{llama_config.vicuna_version}/pytorch_model-00001-of-00002.bin",
                                            f"{llama_config.root_path}{llama_config.vicuna_version}/pytorch_model-00002-of-00002.bin"]
        elif llama_config.vicuna_version == "vicuna-13b-v1.5":
            llama_config.vicuna_weight_path = [f"{llama_config.root_path}{llama_config.vicuna_version}/pytorch_model-00001-of-00003.bin",
                                            f"{llama_config.root_path}{llama_config.vicuna_version}/pytorch_model-00002-of-00003.bin",
                                            f"{llama_config.root_path}{llama_config.vicuna_version}/pytorch_model-00003-of-00003.bin"]
        else:
            raise NotImplementedError
        
        llama_config.llama_dim = llama_config.hidden_size
        llama_config.tokenizer_path = f"{llama_config.root_path}{llama_config.vicuna_version}/"
        llama_config.model_path = f"{llama_config.root_path}{llama_config.vicuna_version}"

        # prepare llama tokenizer(load & and special tokens)
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_config.tokenizer_path, 
                                                            use_fast = False,
                                                            legacy=False)
        
        self.llama_tokenizer.add_tokens([llama_config.seg_token,
                                        llama_config.gs_token,
                                        llama_config.ge_token,
                                        llama_config.vp_token,
                                        ], special_tokens=True)

        self.llama_tokenizer.seg_token_id = self.llama_tokenizer(llama_config.seg_token,add_special_tokens=False)['input_ids'][0]
        self.llama_tokenizer.gs_token_id = self.llama_tokenizer(llama_config.gs_token,add_special_tokens=False)['input_ids'][0]
        self.llama_tokenizer.ge_token_id = self.llama_tokenizer(llama_config.ge_token,add_special_tokens=False)['input_ids'][0]
        self.llama_tokenizer.vp_token_id = self.llama_tokenizer(llama_config.vp_token,add_special_tokens=False)['input_ids'][0]
        self.llama_tokenizer.seg_token = "<seg>"
        self.llama_tokenizer.gs_token = "<gs>"
        self.llama_tokenizer.ge_token = "<ge>"
        self.llama_tokenizer.vp_token = "<vp>" # there is a table ==> there is <vp>
        if llama_config.vicuna_version == "TinyLlama-1.1B-intermediate-step-1195k-token-2.5T":
            self.llama_tokenizer.pad_token_id = self.llama_tokenizer.eos_token_id
        
        # init llama model
        self.llama_model = LLama3dForCausalLM(config=llama_config,
                                        llama_tokenizer=self.llama_tokenizer,
                                        gradient_checkpointing=True)

        # prepare vicuna weight
        if llama_config.load_pretrain_weight:
            vicuna_weight = {}
            assert not llama_config.vicuna_weight_path[0].split(".")[-1] == "safetensor"
            for path in llama_config.vicuna_weight_path:
                weights = torch.load(path, map_location=torch.device('cpu'))
                vicuna_weight.update(weights)
            
            self.llama_model.load_state_dict(vicuna_weight,strict=False)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        # =============== apply lora ===========================
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
    
        lora_target_modules = find_linear_layers(self.llama_model,llama_config.lora_target_modules)

        lora_config = LoraConfig(
            r=llama_config.lora_r,
            lora_alpha=llama_config.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=llama_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llama_model = get_peft_model(self.llama_model, lora_config)
        self.llama_model.print_trainable_parameters()

        # self.llama_model = self.llama_model.to(dtype=torch.float16) # torch.float16 may result in NaN --> torch.bfloat16
        self.llama_model = self.llama_model.bfloat16()
        # froze model
        for name, param in self.llama_model.named_parameters():
            if any(config_item in name for config_item in llama_config.train_layer_list):
                param.requires_grad = True
            else:
                param.requires_grad = False

            # if "lora" in name and "lm_head" in name:
            #     param.requires_grad = True
        
        print_grad_status(self.llama_model)
        if llama_config.use_checkpoint:
            self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    else:
        print(" ====================== llm is disabled ===================")
    if self.global_rank == 0:
        with open(f"{llama_config.save_path}/running_time_llama_conf.pkl", 'wb') as config_file:
            pickle.dump(llama_config, config_file)

    self.llama_config = llama_config