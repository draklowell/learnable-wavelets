from learnable_wavelets.config import load_config
from learnable_wavelets.module import WaveletModule

import torchvision
import torch



def main(config_path: str, state_path: str, input_path: str, grayscale_path: str | None, output_path: str) -> None:
    config = load_config(config_path)
    module = WaveletModule(config)

    module.load_state_dict(torch.load(state_path, map_location="cpu"))
    module.eval()

    image = torchvision.io.read_image(input_path, mode=torchvision.io.ImageReadMode.GRAY)

    if grayscale_path is not None:
        torchvision.io.write_png(image, grayscale_path)

    image = image.unsqueeze(0).float() / (255.0 / 2) - 1
    with torch.inference_mode():
        output = module(image)

    output = (output.squeeze(0).clamp(-1, 1) + 1) * (255.0 / 2)
    output = output.round().byte()
    torchvision.io.write_png(output, output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with a trained wavelet module")
    parser.add_argument("--config", help="Path to the module config file", default="params/config.yaml")
    parser.add_argument("--state", help="Path to the module state file", default="params/state.pt")
    parser.add_argument("--input", required=True, help="Path to the input image")
    parser.add_argument("--grayscale", help="Path to save the input image, but grayscale (png)", default=None)
    parser.add_argument("--output", required=True, help="Path to save the output image (png)")

    args = parser.parse_args()
    main(args.config, args.state, args.input, args.grayscale, args.output)
