import argparse

import torch
import open_clip


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BioMedCLIP concept bank")
    parser.add_argument("--output", type=str, default="concepts.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    concepts = [
        "microaneurysms",
        "hemorrhages",
        "hard exudates",
        "cotton wool spots",
        "normal retina",
    ]

    model, _, preprocess = open_clip.create_model_and_transforms("biomedclip", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("biomedclip")
    model = model.to(args.device)
    model.eval()

    with torch.no_grad():
        tokens = tokenizer(concepts).to(args.device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

    torch.save(text_features.cpu(), args.output)
    print(f"Saved concepts to {args.output}")


if __name__ == "__main__":
    main()
