"""Utilities to train PyTorch models with less boilerplate."""

import collections
import os
import tempfile
import time
import torchvision
import matplotlib.pyplot as plt
import torch
from torch.utils import tensorboard
import numpy as np

pathToCluster = r"/home/dsi/eyalbetzalel/image-gpt/downloads/kmeans_centers.npy"  # TODO : add path to cluster dir
global clusters
clusters = torch.from_numpy(np.load(pathToCluster)).float()


class Trainer:
    """An object which encapsulates the training and evaluation loop.

    Note that the trainer is stateful. This means that calling
    `trainer.continuous_train_and_eval()` a second time will cause training
    to pick back up from where it left off.
    """

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        eval_loader,
        save_checkpoint_epochs=1,
        sample_epochs=1,
        sample_fn=None,
        lr_scheduler=None,
        log_dir=None,
        device="cuda",
        # Hyper parameters :
        n_channels=64,
        n_pixel_snail_blocks=8,
        n_residual_blocks=2,
        attention_value_channels=128,
        attention_key_channels=16,
    ):
        """Initializes a new Trainer instance.

        Args:
            model: Model to train and evaluate.
            loss_fn: A `fn(inputs, targets, predictions)->output`. The output can either
                be a single loss Tensor or a dictionary containing multiple loss
                Tensors. The dictionary must contain a `loss` key which will be used as
                the primary loss for backprop.
            optimizer: Optimizer to use when training.
            train_loader: DataLoader for the training set.
            eval_loader: DataLoader for the evaluation set.
            save_checkpoint_epochs: Number of epochs to wait between checkpoints. Note
                that this does not affect TensorBoard logging frequency.
            sample_epochs: Number of epochs to wait between generating new image samples
                and logging them to TensorBoard. If not `None`, `sample_fn` must be
                provided.
            sample_fn: A `fn(model)->Tensor` which returns an NCHW Tensor of images to
                log to TensorBoard.
            lr_scheduler: An torch.optim.lr_scheduler whose step() method is called
                after every batch.
            log_dir: The directory where to log checkpoints and TensorBoard metrics. If
                `None` a temporary directory is created (note that this directory is not
                cleaned up automatically).
            device: The device to place the model and data. Either string or
                torch.device.
        """
        # Stateful objects that need to be saved.

        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._loss_fn = loss_fn
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        self._save_checkpoint_epochs = save_checkpoint_epochs
        self._device = torch.device(device) if isinstance(device, str) else device
        self._model = model.to('cuda')
        self._sample_epochs = sample_epochs
        self._sample_fn = sample_fn
        # if self._sample_epochs:
        #     msg = "sample_fn cannot be None if sample_epochs is not None"
        #     assert self._sample_fn, msg


        self._step = 0
        self._epoch = 0
        self._examples_processed = 0
        self._time_taken = 0

        # Pass hyper-parameters :

        self.n_channels = n_channels
        self.n_pixel_snail_blocks = n_pixel_snail_blocks
        self.n_residual_blocks = n_residual_blocks
        self.attention_value_channels = attention_value_channels
        self.attention_key_channels = attention_key_channels

        self.hp_str = "ep_" + str(self._epoch) + "_ch_" + str(self.n_channels) + "_psb_" + str(self.n_pixel_snail_blocks) + "_resb_" + \
            str(self.n_residual_blocks) + "_atval_" + str(self.attention_value_channels) + \
            "_attk_" + str(self.attention_key_channels)

        self._log_dir = (log_dir + "/" + self.hp_str) or tempfile.mkdtemp()

        self._summary_writer = tensorboard.SummaryWriter(self._log_dir, max_queue=100)

    def _path(self, file_name):
        return os.path.join(self._log_dir, file_name)

    def _save_checkpoint(self):

        if self._epoch % self._save_checkpoint_epochs != 0:
            return

        hp_str = self.hp_str + "_"

        fname_model = hp_str + "model_state"
        fname_optimizer = hp_str + "optimizer_state"
        fname_lr_scheduler = hp_str + "lr_scheduler_state"
        torch.save(self._model.state_dict(), self._path(fname_model))
        torch.save(self._optimizer.state_dict(), self._path(fname_optimizer))
        if self._lr_scheduler is not None:
            torch.save(
                self._lr_scheduler.state_dict(), self._path(fname_lr_scheduler)
            )
        # TODO(eugenhotaj): Instead of saving these internal counters one at a
        # time, maybe we can save them as a dictionary.
        torch.save(self._step, self._path(hp_str + "step"))
        torch.save(self._epoch, self._path(hp_str + "epoch"))
        torch.save(self._examples_processed, self._path(hp_str + "examples_processed"))
        torch.save(self._time_taken, self._path(hp_str + "time_taken"))

    def load_from_checkpoint(self):
        """Attempts to load Trainer state from the internal log_dir."""
        self._model.load_state_dict(torch.load(self._path("model_state")))
        self._optimizer.load_state_dict(torch.load(self._path("optimizer_state")))
        if self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(
                torch.load(self._path("lr_scheduler_state"))
            )
        self._step = torch.load(self._path("step"))
        self._epoch = torch.load(self._path("epoch"))
        self._examples_processed = torch.load(self._path("examples_processed"))
        self._time_taken = torch.load(self._path("time_taken"))
        # NOTE(eugenhotaj): We need to replace the SummaryWriter and ensure any
        # logs written after the last saved checkpoint are purged.
        self._summary_writer.close()
        self._summary_writer = tensorboard.SummaryWriter(
            self._log_dir, max_queue=100, purge_step=self._step
        )

    def _get_loss_dict(self, loss):
        loss = loss if isinstance(loss, dict) else {"loss": loss}
        assert "loss" in loss, 'Losses dictionary does not contain "loss" key.'
        return loss

    # TODO(eugenhotaj): Consider removing the 'training' argument and just using
    # self.model.parameters().training.
    def _log_loss_dict(self, loss_dict, training):
        for key, loss in loss_dict.items():
            key = key if key == "loss" else f"loss/{key}"
            self._summary_writer.add_scalars(
                key, {"train" if training else "eval": loss}, self._step
            )

    def train_one_batch(self, x, y):
        """Trains the model on a single batch of examples.

        Subclasses can override this method to define custom training loops.
        """
        preds = self._model(x)
        loss = self._loss_fn(x, y, preds)
        return loss

    def _train_one_batch(self, x, y):
        self._model.train()
        x = x.to(self._device)
        if y is not None:
            y = y.to(self._device)
        self._optimizer.zero_grad()
        loss = self._get_loss_dict(self.train_one_batch(x, y))
        loss["loss"].backward()
        self._optimizer.step()
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        return {k: v.item() for k, v in loss.items()}

    def eval_one_batch(self, x, y):
        """Evaluates the model on a single batch of examples.

        Subclasses can override this method to define custom evaluation loops.
        """
        preds = self._model(x)
        loss = self._loss_fn(x, y, preds)
        return loss

    def _eval_one_batch(self, x, y):
        with torch.no_grad():
            self._model.eval()
            x = x.to(self._device)
            if y is not None:
                y = y.to(self._device)
            loss = self._get_loss_dict(self.eval_one_batch(x, y))
            return {k: v.item() for k, v in loss.items()}

    def _sample(self):


        def plot_images_grid(x: torch.tensor, export_img, title: str = '', nrow=8, padding=2, normalize=True,
                             pad_value=0):
            """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""

            grid = torchvision.utils.make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
            npgrid = grid.cpu().numpy()
            im = np.transpose(npgrid, (1, 2, 0))
            plt.imsave(export_img, im)

        ####################################################################################################################
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EB~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sampling while training :
        if self._epoch % self._sample_epochs == 0:

            for i in range(10):
                print("------------------ Sampling " + str(i) + " out of 10 (long) ------------------")
                sample = self._model.sample(out_shape=[1024, 1])
                sample = torch.reshape(sample, [32, 32])
                sample = sample[None, :, :]
                sample = torch.round(127.5 * (clusters[sample.long()] + 1.0))
                sample = sample.permute(0, 3, 1, 2)
                f_name = "./" + self.hp_str + "/samples/sample_epoch_" + str(self._epoch) + "image_" + str(i) + ".png"
                plot_images_grid(sample, f_name)
            self._summary_writer.add_images("sample", sample, self._step)
        ####################################################################################################################

    def interleaved_train_and_eval(self, n_epochs):
        """Trains and evaluates (after each epoch) for n_epochs."""


        for epoch in range(n_epochs):
            start_time = time.time()
            print("------------------ Epoch = " + str(epoch) + " ------------------")
            # Train.
            for i, batch in enumerate(self._train_loader):

                batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
                x, y = batch
                x, y = x.to('cuda'), y.to('cuda')
                self._examples_processed += x.shape[0]
                lrs = {
                    f"group_{i}": param["lr"]
                    for i, param in enumerate(self._optimizer.param_groups)
                }
                self._summary_writer.add_scalars("loss/lr", lrs, self._step)
                loss = self._train_one_batch(x, y)
                self._log_loss_dict(loss, training=True)
                self._time_taken += time.time() - start_time
                start_time = time.time()
                self._summary_writer.add_scalar(
                    "speed/examples_per_sec",
                    self._examples_processed / self._time_taken,
                    self._step,
                )
                self._summary_writer.add_scalar(
                    "speed/millis_per_example",
                    self._time_taken / self._examples_processed * 1000,
                    self._step,
                )
                self._summary_writer.add_scalar(
                    "progress/epoch", self._epoch, self._step
                )
                self._summary_writer.add_scalar("progress/step", self._step, self._step)
                self._step += 1

            # Evaluate

            self._model.eval()
            total_examples, total_loss = 0, collections.defaultdict(int)
            for batch in self._eval_loader:
                batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
                x, y = batch
                x, y = x.to('cuda'), y.to('cuda')
                n_examples = x.shape[0]
                total_examples += n_examples
                for key, loss in self._eval_one_batch(x, y).items():
                    total_loss[key] += loss * n_examples
            loss = {key: loss / total_examples for key, loss in total_loss.items()}
            self._log_loss_dict(loss, training=False)

            self._sample()
            self._save_checkpoint()
            self._epoch += 1

        self._summary_writer.close()
