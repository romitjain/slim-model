import torch
from typing import List, Optional, Tuple
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Config, Qwen2Model, Qwen2ForCausalLM

class PrunedQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx

        delattr(self, 'self_attn')
        delattr(self, 'post_attention_layernorm')

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return (hidden_states,)


class PrunedQwen2Model(Qwen2Model):
    def __init__(self, config: Qwen2Config, pruned_layers: List[int]):
        super().__init__(config)
        self.pruned_layers = pruned_layers

        for layer_idx in self.pruned_layers:
            self.layers[layer_idx] = PrunedQwen2DecoderLayer(config, layer_idx)


class PrunedQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config, pruned_layers: List[int]):
        super().__init__(config)
        self.model = PrunedQwen2Model(config, pruned_layers)
