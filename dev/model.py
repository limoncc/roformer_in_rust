import json
import torch
import transformers
from transformers import RoFormerTokenizerFast, RoFormerModel
transformers.logging.set_verbosity_error()
from typing import  Dict, Tuple

class RoFormer_Doc2vec(torch.nn.Module):
    #
    def __init__(self, model_path: str):
        super().__init__()
        self.roformer = RoFormerModel.from_pretrained(model_path, ignore_mismatched_sizes=True)
        self.pooler = torch.nn.Linear(768, 768)
        self.activation = torch.nn.Tanh()
        ...
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None
    ):
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        pooled_output = self.pooler(cls_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
    @classmethod
    def read_conf(cls, conf_path: str) -> Tuple[Dict, Dict]:
        # 做一些映射
        with open(conf_path) as f:
            conf_dict = json.loads(f.read())
            ...
        param_mapping = {
            'roformer.embeddings.word_embeddings.weight': f'roformer.embeddings.word_embeddings.weight',
            'roformer.embeddings.token_type_embeddings.weight': f'roformer.embeddings.token_type_embeddings.weight',
            'roformer.embeddings.LayerNorm.weight': f'roformer.embeddings.LayerNorm.weight',
            'roformer.embeddings.LayerNorm.bias': f'roformer.embeddings.LayerNorm.bias',
            'roformer.encoder.embed_positions.weight': f'roformer.encoder.embed_positions.weight',
            'pooler.weight': 'pooler.dense.weight',
            'pooler.bias': 'pooler.dense.bias'
            
        }
        for i in range(conf_dict['num_hidden_layers']):
            source_prefix_i = f'roformer.encoder.layer.{i}'
            target_prefix_i = f'roformer.encoder.layer.{i}'
            param_mapping.update({f'{target_prefix_i}.attention.self.query.weight': f'{source_prefix_i}.attention.self.query.weight',
                                  f'{target_prefix_i}.attention.self.query.bias': f'{source_prefix_i}.attention.self.query.bias',
                                  f'{target_prefix_i}.attention.self.key.weight': f'{source_prefix_i}.attention.self.key.weight',
                                  f'{target_prefix_i}.attention.self.key.bias': f'{source_prefix_i}.attention.self.key.bias',
                                  f'{target_prefix_i}.attention.self.value.weight': f'{source_prefix_i}.attention.self.value.weight',
                                  f'{target_prefix_i}.attention.self.value.bias': f'{source_prefix_i}.attention.self.value.bias',
                                  f'{target_prefix_i}.attention.output.dense.weight': f'{source_prefix_i}.attention.output.dense.weight',
                                  f'{target_prefix_i}.attention.output.dense.bias': f'{source_prefix_i}.attention.output.dense.bias',
                                  f'{target_prefix_i}.attention.output.LayerNorm.weight': f'{source_prefix_i}.attention.output.LayerNorm.weight',
                                  f'{target_prefix_i}.attention.output.LayerNorm.bias': f'{source_prefix_i}.attention.output.LayerNorm.bias',
                                  f'{target_prefix_i}.intermediate.dense.weight': f'{source_prefix_i}.intermediate.dense.weight',
                                  f'{target_prefix_i}.intermediate.dense.bias': f'{source_prefix_i}.intermediate.dense.bias',
                                  f'{target_prefix_i}.output.dense.weight': f'{source_prefix_i}.output.dense.weight',
                                  f'{target_prefix_i}.output.dense.bias': f'{source_prefix_i}.output.dense.bias',
                                  f'{target_prefix_i}.output.LayerNorm.weight': f'{source_prefix_i}.output.LayerNorm.weight',
                                  f'{target_prefix_i}.output.LayerNorm.bias': f'{source_prefix_i}.output.LayerNorm.bias'
                                  })
            ...
        
        return conf_dict, param_mapping
        ...
    
    @classmethod
    def load(cls, model_path):
        conf_dict, param_mapping = cls.read_conf(f"{model_path}/config.json")
        source_params = torch.load(f"{model_path}/pytorch_model.bin")
        
        model = cls(model_path)
        target_params = model.state_dict()
        
        for target, source in param_mapping.items():
            target_params[target] = source_params[source]
            ...
        
        model.load_state_dict(target_params)
        del source_params, target_params
        
        return model
        ...
    
    ...


path = "./data/model/roformer_chinese_sim_char_base"
tokenizer = RoFormerTokenizerFast.from_pretrained(path)
model = RoFormer_Doc2vec.load(path)
model.eval()
inputs = tokenizer(["我爱你", "你爱我", "rust是ai应用开发第一语言"], padding=True, truncation=True, max_length=3000, return_tensors="pt")
outputs = model(**inputs)
trace_model = torch.jit.trace(model, input)
trace_model.save("./data/doc2vec.jit")
model_jit = torch.jit.load("./data/model.jit")
outputs_jit = model_jit(**inputs)