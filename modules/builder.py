import torch
from transformers import ViTConfig, AutoConfig
from utils.func import print_msg

from .bridge import FineGrainedPromptTuning, FusionModule
from .side_vit import ViTForImageClassification as SideViT
# 引用我们刚刚重写的通用 FrozenViT
from .frozen_vit import FrozenViT


def generate_model(cfg):
    model = build_model(cfg)
    model = model.to(cfg.base.device)

    # 如果启用了预加载 (Preload)，就不需要再加载冻结模型了，直接返回 None
    if cfg.dataset.preload_path:
        frozen_encoder = None
    else:
        # 如果是实时训练 (无预加载)，则需要构建冻结编码器
        frozen_encoder = build_frozen_encoder(cfg).to(cfg.base.device)

        # 打印参数统计信息
        num_learnable_params = 0
        total_params = 0
        for _, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                num_learnable_params += param.numel()

        if frozen_encoder is not None:
            for _, param in frozen_encoder.named_parameters():
                total_params += param.numel()
                # frozen_encoder 应该是完全冻结的，requires_grad 应该都是 False

        print('Total params: {}'.format(total_params))
        print('Learnable params: {}'.format(num_learnable_params))
        if total_params > 0:
            print('Learnable params ratio: {:.4f}%'.format(num_learnable_params / total_params * 100))

    return frozen_encoder, model


def load_weights(model, checkpoint):
    # 增加 weights_only=False 警告的兼容性处理（虽然只是打印信息）
    try:
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    except:
        weights = torch.load(checkpoint, map_location='cpu')

    model.load_state_dict(weights)
    print_msg('Load weights form {}'.format(checkpoint))


def build_model(cfg):
    # 构建 Side Network (可训练的小网络)
    num_layers = cfg.network.layers_to_extract.count('-') + 1 + cfg.network.layers_to_extract.count(',')
    side_dimension = 768 // cfg.network.side_reduction_ratio  # 假设基础维度是 768 (ViT-Base)
    prompts_dim = 768 // cfg.network.prompt_reduction_ratio

    out_features = select_out_features(cfg.dataset.name, cfg.dataset.num_classes)

    # 构建融合模块
    fusion_module = FusionModule(
        dim=side_dimension,
        num_prompts=cfg.network.num_prompts,
        prompt_dim=prompts_dim,
        prompt_norm=cfg.network.prompt_norm,
        prompt_proj=cfg.network.prompt_proj
    )

    # Side Network 仍然使用 ViT 的配置结构
    side_config = ViTConfig.from_pretrained(
        cfg.network.pretrained_path,
        num_hidden_layers=num_layers,
        hidden_size=side_dimension,
        intermediate_size=side_dimension * 4,
        image_size=cfg.network.side_input_size,
        num_labels=out_features,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0
    )
    side_encoder = SideViT(side_config)

    model = FineGrainedPromptTuning(side_encoder, fusion_module)
    return model


def build_frozen_encoder(cfg):
    """
    构建冻结的预训练编码器 (LPM)。
    现在使用通用的 FrozenViT 类，支持 DINOv3 等各种模型。
    """
    print(f"Loading Frozen Encoder from local path: {cfg.network.pretrained_path}")

    # 直接加载，无需手动配置 config 的细节，让 HF 自动处理
    frozen_encoder = FrozenViT.from_pretrained(
        cfg.network.pretrained_path,
        output_hidden_states=True
    )

    # 再次确保冻结
    frozen_encoder.eval()
    for p in frozen_encoder.parameters():
        p.requires_grad = False

    return frozen_encoder


def parse_layers(layers_to_extract):
    if '-' in layers_to_extract:
        start, end = layers_to_extract.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return [int(x) for x in layers_to_extract.split(',')]