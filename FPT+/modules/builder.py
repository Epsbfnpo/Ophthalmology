import torch
from transformers import ViTConfig, AutoConfig
from utils.func import print_msg, select_out_features

from .bridge import FineGrainedPromptTuning, FusionModule
from .side_vit import ViTForImageClassification as SideViT
from .frozen_vit import FrozenViT


def generate_model(cfg):
    model = build_model(cfg)
    model = model.to(cfg.base.device)

    if cfg.dataset.preload_path:
        frozen_encoder = None
    else:
        frozen_encoder = build_frozen_encoder(cfg).to(cfg.base.device)

        num_learnable_params = 0
        total_params = 0
        for _, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                num_learnable_params += param.numel()

        if frozen_encoder is not None:
            for _, param in frozen_encoder.named_parameters():
                total_params += param.numel()

        print('Total params: {}'.format(total_params))
        print('Learnable params: {}'.format(num_learnable_params))
        if total_params > 0:
            print('Learnable params ratio: {:.4f}%'.format(num_learnable_params / total_params * 100))

    return frozen_encoder, model


def load_weights(model, checkpoint):
    try:
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    except:
        weights = torch.load(checkpoint, map_location='cpu')

    model.load_state_dict(weights)
    print_msg('Load weights form {}'.format(checkpoint))


def build_model(cfg):
    layers_list = parse_layers(cfg.network.layers_to_extract)
    num_layers = len(layers_list)

    base_dim = 768

    side_dimension = base_dim // cfg.network.side_reduction_ratio

    prompts_dim = base_dim // cfg.network.prompt_reduction_ratio

    out_features = select_out_features(cfg.dataset.num_classes, cfg.train.criterion)

    fusion_module = FusionModule(
        num_layers=num_layers,
        in_dim=base_dim,
        out_dim=side_dimension,
        num_heads=12,
        num_prompts=cfg.network.num_prompts,
        prompt_dim=prompts_dim,
        prompt_norm=cfg.network.prompt_norm,
        prompt_proj=cfg.network.prompt_proj
    )

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

    model = FineGrainedPromptTuning(side_encoder, fusion_module, layer_indices=layers_list)

    return model


def build_frozen_encoder(cfg):
    print(f"Loading Frozen Encoder from local path: {cfg.network.pretrained_path}")

    frozen_encoder = FrozenViT.from_pretrained(
        cfg.network.pretrained_path,
        output_hidden_states=True
    )

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