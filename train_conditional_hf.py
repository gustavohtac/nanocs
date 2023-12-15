import argparse
from types import SimpleNamespace
import wandb
from utils import *
from diffusion_conditional_hf import Diffusion

config = SimpleNamespace(    
    run_name = "DDPM_conditional_hf",
    epochs = 20,
    noise_steps=1000,
    seed = 42,
    batch_size = 2,
    img_size = 128,
    channels = 1,
    dataset_path = "data",
    train_folder = "multicoil_train",
    val_folder = "multicoil_train_single",
    model_ckpt='./models/DDPM_conditional/ckpt_9.pt',
    ema_model_ckpt='./models/DDPM_conditional/ema_ckpt_9.pt',
    device = "cuda",
    do_validation = True,
    fp16 = True,
    log_every_epoch = 1,
    num_workers=6,
    lr = 1e-4
)


def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    diffuser = Diffusion(
        config.noise_steps, 
        img_size=config.img_size, 
        batch_size=config.batch_size,
        channels=config.channels, 
        run_name=config.run_name,
        model_ckpt=config.model_ckpt,
        ema_model_ckpt=config.ema_model_ckpt
    )
    with wandb.init(project="train_sd", group="train", config=config):
        diffuser.prepare(config)
        diffuser.fit(config)