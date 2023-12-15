from utils import *
from diffusers import UNet2DModel
from diffusion import Diffusion

def sample(args):

    device = args.device
    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=args.channels,
        out_channels=args.channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    model.load_state_dict(torch.load(args.pretrained_model_path))

    diffusion = Diffusion(img_size=args.image_size, device=device)
    sampled_images = diffusion.sample(model, n=9)
    image_grid = make_grid(sampled_images, rows=3, cols=3)
    image_grid.save(os.path.join("results", args.run_name, f"sample_{args.output_file}.png"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.channels = 1
    args.output_file = 't1'
    args.image_size = 320
    args.pretrained_model_path = "./models/DDPM_Uncondtional/ckpt.pt"
    args.device = "cuda"
    sample(args)


if __name__ == '__main__':
    launch()