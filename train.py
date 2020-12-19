
"""Main training script for models."""

import argparse

import torch
from torch import distributions
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import transforms

import pytorch_generative as pg


MODEL_DICT = {
    "gated_pixel_cnn": pg.models.gated_pixel_cnn,
    "image_gpt": pg.models.image_gpt,
    "made": pg.models.made,
    "nade": pg.models.nade,
    "pixel_cnn": pg.models.pixel_cnn,
    "pixel_snail": pg.models.pixel_snail,
    "vae": pg.models.vae,
    "vq_vae": pg.models.vq_vae,
    "vq_vae_2": pg.models.vq_vae_2,
}


def main(args):
    device = "cuda" if args.use_cuda else "cpu"
    MODEL_DICT[args.model].reproduce(
        args.n_epochs, args.batch_size, args.log_dir, device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="the available models to train",
        default="pixel_snail",
        choices=list(MODEL_DICT.keys()),
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        help="number of training epochs",
        default=100
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="the training and evaluation batch_size",
        default=128,
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="the directory where to log data",
        default="./tmp/run",
    )
    parser.add_argument("--use-cuda", help="whether to use CUDA", action="store_true")
    args = parser.parse_args()

    main(args)
