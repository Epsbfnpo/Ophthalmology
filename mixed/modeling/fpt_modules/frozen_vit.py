import torch
import torch.nn as nn
import inspect
from transformers import AutoModel, AutoConfig, PreTrainedModel


class FrozenViT(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, interpolate_pos_encoding=False, **kwargs):
        forward_kwargs = {"output_hidden_states": True, "return_dict": True}

        if hasattr(self.model, "forward"):
            sig = inspect.signature(self.model.forward)
            if "interpolate_pos_encoding" in sig.parameters:
                forward_kwargs["interpolate_pos_encoding"] = interpolate_pos_encoding

        outputs = self.model(x, **forward_kwargs)

        features = torch.stack(outputs.hidden_states)

        return None, features, features

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        if "output_hidden_states" in kwargs:
            config.output_hidden_states = kwargs.pop("output_hidden_states")

        model = cls(config)

        pretrained_backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            *model_args,
            **kwargs
        )
        model.model = pretrained_backbone

        return model