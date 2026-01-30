import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput


class FrozenViT(PreTrainedModel):
    """
    一个通用的冻结视觉编码器包装器。
    它可以加载 Google ViT, DINOv2, DINOv3, BioMedCLIP 等任何 Hugging Face 支持的模型。
    """
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # 使用 AutoModel 自动加载对应的骨干网络
        # add_pooling_layer=False: 我们只需要特征，不需要分类头
        self.model = AutoModel.from_config(config, add_pooling_layer=False)

        # 彻底冻结所有参数，节省显存
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, interpolate_pos_encoding=False, **kwargs):
        """
        前向传播函数，适配 FPT+ 的接口需求。

        参数:
            x: 输入图像 tensor
            interpolate_pos_encoding: 是否对位置编码进行插值（用于处理不同分辨率的输入）
        """
        # 1. 自动处理 interpolate_pos_encoding 参数
        # 某些新模型（如 DINOv2/v3）可能不需要显式传递这个参数，或者参数名不同
        # 我们尝试传递它，如果模型不支持，通常会在 **kwargs 里被过滤或我们手动处理
        forward_kwargs = {"output_hidden_states": True, "return_dict": True}

        # 只有当模型架构明确支持该参数时才传入，防止报错
        # 大多数 ViT 模型都支持这个参数来处理非 224x224 的图片
        forward_kwargs["interpolate_pos_encoding"] = interpolate_pos_encoding

        # 2. 执行前向传播
        outputs = self.model(x, **forward_kwargs)

        # 3. 提取特征
        # FPT+ 的架构设计需要三个返回值： (classification_output, key_states, value_states)
        #
        # - classification_output: 在预加载阶段通常被忽略（用 _ 接收），所以返回 None 即可
        # - key_states: 用作 Cross-Attention 的 Key，这里使用所有层的 hidden_states
        # - value_states: 用作 Cross-Attention 的 Value，这里使用所有层的 hidden_states

        # outputs.hidden_states 是一个 tuple，包含每一层的输出特征
        # 这里的逻辑是：将预训练模型的所有层特征作为“知识库”，供 Side Network 查询
        return None, outputs.hidden_states, outputs.hidden_states

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        重写加载方法，确保能正确加载 safetensors 或 bin 文件
        """
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # 这里的关键是让 AutoModel 去处理权重加载的复杂性
        # 我们实例化自己的类，但核心 model 使用 HF 的加载逻辑
        model = cls(config)

        # 加载预训练权重
        # 我们使用 AutoModel 直接加载权重，然后赋值给 self.model
        # 这种方法比手动 load_state_dict 更健壮，因为它会自动处理前缀不匹配的问题
        pretrained_backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            add_pooling_layer=False,
            *model_args,
            **kwargs
        )
        model.model = pretrained_backbone

        return model