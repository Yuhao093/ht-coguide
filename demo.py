"""Minimal single-pair demonstration of the guided perturbation."""
import argparse

import torch
import torchvision
from PIL import Image
from torchvision import transforms

from models.model import CLIPEncoder, Decoder
from priority import MODES, PriorityConfig, allocate
from util import load_decoder


def load_image(path, device):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return transform(image).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=16 / 255)
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--decoder_path", type=str, required=True)
    parser.add_argument("--clean_image_path", type=str, required=True)
    parser.add_argument("--target_image_path", type=str, required=True)
    parser.add_argument("--priority_mode", type=str, default="triple", choices=list(MODES))
    parser.add_argument("--topk_ratio", type=float, default=0.35)
    parser.add_argument("--low_region_weight", type=float, default=0.385)
    parser.add_argument("--w_noise", type=float, default=0.45)
    parser.add_argument("--w_host", type=float, default=0.30)
    parser.add_argument("--w_target", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", type=str, default="output.png")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = CLIPEncoder(args.model_name, device=device).to(device).eval()
    decoder = Decoder(embed_dim=512)
    load_decoder(args.decoder_path, decoder, device=device)
    decoder = decoder.to(device).eval()

    host = load_image(args.clean_image_path, device)
    target = load_image(args.target_image_path, device)
    config = PriorityConfig(epsilon=args.eps, ratio=args.topk_ratio,
                            low_region_weight=args.low_region_weight, w_noise=args.w_noise,
                            w_host=args.w_host, w_target=args.w_target, mode=args.priority_mode)
    with torch.no_grad():
        raw = decoder(encoder.encode_img(target))
    result = allocate(raw, host, target, encoder, config)
    torchvision.utils.save_image(result["adversarial"][0], args.output_path)
    print(f"Saved adversarial image to {args.output_path}")


if __name__ == "__main__":
    main()
