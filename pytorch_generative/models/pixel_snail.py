"""Implementation of PixelSNAIL [1].

PixelSNAIL extends PixelCNN [2] (and its variants) by introducing a causally
masked attention layer. This layer extends the model's receptive field by 
allowing each pixel to explicitly depend on all previous pixels. PixelCNN's
receptive field, on the other hand, can only be increased by using deeper 
networks. The attention block also naturally resolves the blind spot in PixelCNN
without needing a complex two stream architecture.

NOTE: Unlike [1], we use skip connections from each PixelSNAILBlock to the 
output. We find that this greatly stabilizes the model during training and gets
rid of exploding gradient issues. It also massively speeds up convergence.

References (used throughout the code):
    [1]: https://arxiv.org/abs/1712.09763
    [2]: https://arxiv.org/abs/1606.05328
"""

import torch
from torch import distributions
from torch import nn
from torch.nn import functional as F
from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base
import numpy as np

pathToCluster = r"/home/dsi/eyalbetzalel/image-gpt/downloads/kmeans_centers.npy"  # TODO : add path to cluster dir
global clusters
clusters = torch.from_numpy(np.load(pathToCluster)).float()

def _elu_conv_elu(conv, x):
    return F.elu(conv(F.elu(x)))

class ResidualBlock(nn.Module):
    """Residual block with a gated activation function."""

    def __init__(self, n_channels):
        """Initializes a new ResidualBlock.

        Args:
            n_channels: The number of input and output channels.
        """
        super().__init__()
        self._input_conv = nn.Conv2d(
            in_channels=n_channels, out_channels=n_channels, kernel_size=2, padding=1
        )
        self._output_conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            kernel_size=2,
            padding=1,
        )
        self._activation = pg_nn.GatedActivation(activation_fn=nn.Identity())

    def forward(self, x):
        _, c, h, w = x.shape
        out = _elu_conv_elu(self._input_conv, x)[:, :, :h, :w]
        out = self._activation(self._output_conv(out)[:, :, :h, :w])
        return x + out

class PixelSNAILBlock(nn.Module):
    """Block comprised of a number of residual blocks plus one attention block.

    Implements Figure 5 of [1].
    """

    def __init__(
        self,
        n_channels,
        input_img_channels=1,
        n_residual_blocks=2,
        attention_key_channels=4,
        attention_value_channels=32,
    ):
        """Initializes a new PixelSnailBlock instance.

        Args:
            n_channels: Number of input and output channels.
            input_img_channels: The number of channels in the original input_img. Used
                for the positional encoding channels and the extra channels for the key
                and value convolutions in the attention block.
            n_residual_blocks: Number of residual blocks.
            attention_key_channels: Number of channels (dims) for the attention key.
            attention_value_channels: Number of channels (dims) for the attention value.
        """
        super().__init__()

        def conv(in_channels):
            return nn.Conv2d(in_channels, out_channels=n_channels, kernel_size=1)

        self._residual = nn.Sequential(
            *[ResidualBlock(n_channels) for _ in range(n_residual_blocks)]
        )
        self._attention = pg_nn.MaskedAttention(
            in_channels=n_channels + 2 * input_img_channels,
            embed_channels=attention_key_channels,
            out_channels=attention_value_channels,
            is_causal=True,
            extra_input_channels=input_img_channels,
        )
        self._residual_out = conv(n_channels)
        self._attention_out = conv(attention_value_channels)
        self._out = conv(n_channels)

    def forward(self, x, input_img):
        """Computes the forward pass.

        Args:
            x: The input.
            input_img: The original image only used as input to the attention blocks.
        Returns:
            The result of the forward pass.
        """
        res = self._residual(x)
        pos = pg_nn.image_positional_encoding(input_img.shape).to(res.device)
        attn = self._attention(torch.cat((pos, res), dim=1), input_img)
        res, attn = (
            _elu_conv_elu(self._residual_out, res),
            _elu_conv_elu(self._attention_out, attn),
        )
        return _elu_conv_elu(self._out, res + attn)

class PixelSNAIL(base.AutoregressiveModel):
    """The PixelSNAIL model.

    Unlike [1], we implement skip connections from each block to the output.
    We find that this makes training a lot more stable and allows for much faster
    convergence.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_channels=64,
        n_pixel_snail_blocks=8,
        n_residual_blocks=2,
        attention_key_channels=4,
        attention_value_channels=32,
        sample_fn=None,
    ):
        """Initializes a new PixelSNAIL instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output_channels.
            n_channels: Number of channels to use for convolutions.
            n_pixel_snail_blocks: Number of PixelSNAILBlocks.
            n_residual_blocks: Number of ResidualBlock to use in each PixelSnailBlock.
            attention_key_channels: Number of channels (dims) for the attention key.
            attention_value_channels: Number of channels (dims) for the attention value.
            sample_fn: See the base class.
        """
        super().__init__(sample_fn)
        self._input = pg_nn.MaskedConv2d(
            is_causal=True,
            in_channels=in_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
        )
        self._pixel_snail_blocks = nn.ModuleList(
            [
                PixelSNAILBlock(
                    n_channels=n_channels,
                    input_img_channels=in_channels,
                    n_residual_blocks=n_residual_blocks,
                    attention_key_channels=attention_key_channels,
                    attention_value_channels=attention_value_channels,
                )
                for _ in range(n_pixel_snail_blocks)
            ]
        )
        self._output = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels // 2, kernel_size=1
            ),
            nn.Conv2d(
                in_channels=n_channels // 2, out_channels=out_channels, kernel_size=1
            ),
        )

        self._logsoftmax = nn.LogSoftmax(dim = 1)

    def forward(self, x, sampleFlag = False):

        ####################################################################################################################
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EB~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Mapping function from 1-ch cluster to 3-ch RGB images :

        # TODO : When sampling there is bug with the input dimensions to the input function.
        # TODO : Make sure that input dimensions are allign both in sampling and in training.
        # TODO : Add Note of the dimensions in each line in training and debug and compere on sampling

        # Training --> x.shape =  torch.Size([128, 1024])
        # Sampling --> x.shape = torch.Size([1024, 1])

        x = torch.round(127.5 * (clusters[x.long()] + 1.0))

        # Training --> x.shape = torch.Size([128, 1024, 3])
        # Sampling --> x.shape = torch.Size([1024, 1, 3])

        if sampleFlag:
            x = x.permute(1, 0, 2)

        # Sampling --> x.shape = torch.Size([1, 1024, 3])

        x = x[:,:,None,:]

        # Training --> x.shape = torch.Size([128, 1024, 1, 3])
        # Sampling --> x.shape = torch.Size([1, 1024, 1, 3])

        x = torch.reshape(x, [x.shape[0], 32, 32,x.shape[3]])

        # Training --> x.shape = torch.Size([128, 32, 32, 3])
        # Sampling --> x.shape = torch.Size([1, 32, 32, 3])

        x = x.permute(0, 3, 1, 2)

        # Training --> x.shape = torch.Size([128, 3, 32, 32])
        # Sampling --> x.shape =torch.Size([1, 3, 32, 32])

        x = x.to('cuda')
        ####################################################################################################################

        input_img = x
        x = self._input(x)

        for block in self._pixel_snail_blocks:
            x = x + block(x, input_img)
            output = self._output(x)

        output = torch.reshape(output,[x.shape[0],512,-1])


        return self._logsoftmax(output)

def reproduce(n_epochs=457,
              batch_size=128,
              log_dir="/tmp/run",
              device="cuda",
              n_channels=1,
              n_pixel_snail_blocks=1,
              n_residual_blocks=1,
              attention_value_channels = 1,
              attention_key_channels = 1,
              evalFlag = False,
              evaldir = "/tmp/run",
              sampling_part = 1):

    """Training script with defaults to reproduce results.

    The code inside this function is self contained and can be used as a top level
    training script, e.g. by copy/pasting it into a Jupyter notebook.

    Args:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size to use for training and evaluation.
        log_dir: Directory where to log trainer state and TensorBoard summaries.
        device: Device to train on (either 'cuda' or 'cpu').
        debug_loader: Debug DataLoader which replaces the default training and
            evaluation loaders if not 'None'. Do not use unless you're writing unit
            tests.
    """
    from torch import optim
    from torch.nn import functional as F
    from torch.optim import lr_scheduler
    from torch.utils import data
    from torchvision import datasets
    from torchvision import transforms

    from pytorch_generative import trainer
    from pytorch_generative import models

    ####################################################################################################################
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EB~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load ImageGPT Data :

    import gmpm

    train = gmpm.train
    test = gmpm.test


    train_loader = data.DataLoader(
        data.TensorDataset(torch.Tensor(train),torch.rand(len(train))),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    test_loader = data.DataLoader(
        data.TensorDataset(torch.Tensor(test),torch.rand(len(test))),
        batch_size=batch_size,
        num_workers=8,
    )
    attention_value_channels = attention_value_channels
    attention_key_channels = attention_key_channels

    model = models.PixelSNAIL(
        ####################################################################################################################
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EB~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Change Input / Output size :

        # 3 channels - image after clusters mapping function as input to NN :
        in_channels=3,

        # 512 channels - each pixel get probability to get value from 0 to 511
        out_channels=512,

        ####################################################################################################################

        n_channels=n_channels,
        n_pixel_snail_blocks=n_pixel_snail_blocks,
        n_residual_blocks=n_residual_blocks,
        attention_value_channels=attention_value_channels,
        attention_key_channels=attention_key_channels,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 0.999977)

    def loss_fn(x, _, preds):

        ####################################################################################################################
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EB~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update loss function to CrossEntropyLoss :

        x = x.long()
        criterion = nn.NLLLoss()
        B, C, D = preds.size()
        preds_2d = preds.view(B, C, D, -1)
        x_2d = x.view(B, D, -1)
        loss = criterion(preds_2d, x_2d.long())

        ####################################################################################################################

        return loss

    _model = model.to(device)
    trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=test_loader,
        lr_scheduler=scheduler,
        log_dir=log_dir,
        device=device,
        sample_epochs = 5,
        sample_fn=None,
        n_channels=n_channels,
        n_pixel_snail_blocks=n_pixel_snail_blocks,
        n_residual_blocks=n_residual_blocks,
        attention_value_channels=attention_value_channels,
        attention_key_channels=attention_key_channels,
        evalFlag=evalFlag,
        evaldir=evaldir,
        sampling_part=sampling_part
    )

    trainer.interleaved_train_and_eval(n_epochs)
