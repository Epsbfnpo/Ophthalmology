import argparse

import open_clip
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BioMedCLIP concept bank")
    parser.add_argument("--output", type=str, default="concepts.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    l1_concepts = [
        "red lesions in retina",
        "bright lesions in retina",
    ]

    l2_concepts = [
        "microaneurysms",
        "retinal hemorrhages",
        "hard exudates",
        "cotton wool spots",
    ]

    print(f"Generating embeddings for {len(l1_concepts)} L1 concepts and {len(l2_concepts)} L2 concepts...")

    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, _, _ = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(args.device)
    model.eval()

    all_concepts = l1_concepts + l2_concepts

    with torch.no_grad():
        tokens = tokenizer(all_concepts).to(args.device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

    torch.save(text_features.cpu(), args.output)
    print(f"Saved concept bank shape: {text_features.shape} to {args.output}")


if __name__ == "__main__":
    main()
