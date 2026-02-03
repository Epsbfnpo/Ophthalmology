import torch
import torch.nn as nn
import inspect
from transformers import AutoModel, AutoConfig, PreTrainedModel
# [引入 PEFT]
from peft import get_peft_model, LoraConfig


class FrozenViT(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)

        # 1. 初始状态：冻结所有参数
        # 之后 LoRA 会自动将适配器部分的参数设为可训练
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, interpolate_pos_encoding=False, **kwargs):
        # -----------------------------------------------------------
        # 1. Embeddings 处理
        # -----------------------------------------------------------
        # 兼容 PEFT 包装后的属性访问
        embeddings_module = None
        if hasattr(self.model, "embeddings"):
            embeddings_module = self.model.embeddings
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model.model, "embeddings"):
            embeddings_module = self.model.base_model.model.embeddings

        if embeddings_module is not None:
            embed_kwargs = {}
            if hasattr(embeddings_module, "forward"):
                sig = inspect.signature(embeddings_module.forward)
                if "interpolate_pos_encoding" in sig.parameters:
                    embed_kwargs["interpolate_pos_encoding"] = interpolate_pos_encoding

            hidden_states = embeddings_module(x, **embed_kwargs)
        else:
            raise AttributeError(
                "Model missing 'embeddings' module. (Checked self.model and self.model.base_model.model)")

        # -----------------------------------------------------------
        # 2. RoPE 计算 (DINOv3 特有)
        # -----------------------------------------------------------
        position_embeddings = None

        rope_module = None
        if hasattr(self.model, "rope_embeddings"):
            rope_module = self.model.rope_embeddings
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model.model, "rope_embeddings"):
            rope_module = self.model.base_model.model.rope_embeddings

        if rope_module is not None:
            # DINOv3 修复：传入原始图像 x 用于计算分辨率相关的 RoPE
            position_embeddings = rope_module(x)

        # -----------------------------------------------------------
        # 3. 获取 Transformer Layers
        # -----------------------------------------------------------
        layers = None
        if hasattr(self.model, "layer"):
            layers = self.model.layer
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model.model, "layer"):
            layers = self.model.base_model.model.layer
        elif hasattr(self.model, "layers"):
            layers = self.model.layers

        if layers is None:
            raise AttributeError("Could not find 'layer' or 'layers' in model.")

        # -----------------------------------------------------------
        # 4. 手动前向传播 (支持 LoRA)
        # -----------------------------------------------------------
        all_hidden_states = []
        all_hidden_states.append(hidden_states)

        last_attention_map = None

        for i, layer_module in enumerate(layers):
            is_last_layer = (i == len(layers) - 1)

            layer_kwargs = {}
            if hasattr(layer_module, "forward"):
                sig = inspect.signature(layer_module.forward)

                # 传递 output_attentions
                if "output_attentions" in sig.parameters:
                    layer_kwargs["output_attentions"] = is_last_layer

                # 传递 position_embeddings (RoPE)
                if "position_embeddings" in sig.parameters and position_embeddings is not None:
                    layer_kwargs["position_embeddings"] = position_embeddings

            # 执行 Layer Forward (LoRA 权重会自动生效)
            layer_outputs = layer_module(hidden_states, **layer_kwargs)

            # 处理返回值
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                if is_last_layer and len(layer_outputs) > 1:
                    last_attention_map = layer_outputs[1]
            else:
                hidden_states = layer_outputs

            all_hidden_states.append(hidden_states)

        # -----------------------------------------------------------
        # 5. Final Norm
        # -----------------------------------------------------------
        norm_module = None
        if hasattr(self.model, "norm"):
            norm_module = self.model.norm
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model.model, "norm"):
            norm_module = self.model.base_model.model.norm
        elif hasattr(self.model, "layernorm"):
            norm_module = self.model.layernorm

        if norm_module is not None:
            hidden_states = norm_module(hidden_states)

        # -----------------------------------------------------------
        # 6. 组装返回结果
        # -----------------------------------------------------------
        features = torch.stack(all_hidden_states)
        attentions = [None] * (len(layers) - 1) + [last_attention_map]

        return None, features, tuple(attentions)

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

        # =========================================================
        # [核心修改] 注入 LoRA 适配器 (根据脚本输出的真实名称)
        # =========================================================
        print(f"Injecting LoRA adapters into FrozenViT (Backbone)...")

        # 依据脚本输出：
        # Attention: q_proj, k_proj, v_proj, o_proj
        # MLP: up_proj, down_proj
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=[],
        )

        # 包装模型
        model.model = get_peft_model(model.model, peft_config)

        # 打印可训练参数量 (验证 LoRA 是否生效)
        print(">> Backbone LoRA Status:")
        model.model.print_trainable_parameters()

        return model