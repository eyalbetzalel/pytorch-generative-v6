"""Utilities to train PyTorch models with less boilerplate."""

import collections
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils import data
from torch.utils import tensorboard
from torchvision import datasets
from torchvision import transforms
import pickle

pathToCluster = r"/home/dsi/eyalbetzalel/image-gpt/downloads/kmeans_centers.npy"  # TODO : add path to cluster dir
global clusters
clusters = torch.from_numpy(np.load(pathToCluster)).float()

global clusters_np
clusters_np = np.load(pathToCluster)


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
            save_checkpoint_epochs=5,
            sample_epochs=25,
            sample_fn=None,
            lr_scheduler=None,
            log_dir=None,
            device="cuda",
            # Hyper parameters :
            n_channels=2,
            n_pixel_snail_blocks=2,
            n_residual_blocks=2,
            attention_value_channels=2,
            attention_key_channels=2,
            evalFlag=False,
            evaldir=None,
            sampling_part=1
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
        self.hp_str = "ep_" + str(self._epoch) + "_ch_" + str(self.n_channels) + "_psb_" + str(
            self.n_pixel_snail_blocks) + "_resb_" + \
                      str(self.n_residual_blocks) + "_atval_" + str(self.attention_value_channels) + \
                      "_attk_" + str(self.attention_key_channels)
        self._log_dir = log_dir  # (log_dir + "/" + self.hp_str) # or tempfile.mkdtemp()
        self._summary_writer = tensorboard.SummaryWriter(self._log_dir, max_queue=100)
        self.evalFlag = evalFlag
        self.evaldir = evaldir
        self.sampling_part = sampling_part

    def _path(self, file_name):
        return os.path.join(self._log_dir, file_name)

    def _save_checkpoint(self):

        # if self._epoch % self._save_checkpoint_epochs != 0:
        #    return

        hp_str = self.hp_str + "_epoch_" + str(self._epoch) + "_"

        fname_model = hp_str + "model_state_single_batch"
        fname_optimizer = hp_str + "optimizer_state_single_batch"
        fname_lr_scheduler = hp_str + "lr_scheduler_state_single_batch"
        torch.save(self._model.state_dict(), self._path(fname_model))
        torch.save(self._optimizer.state_dict(), self._path(fname_optimizer))
        if self._lr_scheduler is not None:
            torch.save(
                self._lr_scheduler.state_dict(), self._path(fname_lr_scheduler)
            )
        # TODO(eugenhotaj): Instead of saving these internal counters one at a
        # time, maybe we can save them as a dictionary.
        torch.save(self._step, self._path(hp_str + "step_single_batch"))
        torch.save(self._epoch, self._path(hp_str + "epoch_single_batch"))
        torch.save(self._examples_processed, self._path(hp_str + "examples_processed_single_batch"))
        torch.save(self._time_taken, self._path(hp_str + "time_taken_single_batch"))

    def load_from_checkpoint(self):
        
        hp_str = self.hp_str + "_"

        fname_model = hp_str + "model_state"
        fname_optimizer = hp_str + "optimizer_state"
        fname_lr_scheduler = hp_str + "lr_scheduler_state"
        
        """Attempts to load Trainer state from the internal log_dir."""
        
        self._model.load_state_dict(torch.load(self._path(fname_model)))
        self._optimizer.load_state_dict(torch.load(self._path(fname_optimizer)))
        if self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(
                torch.load(self._path(fname_lr_scheduler))
            )
        self._step = torch.load(self._path(hp_str + "step"))
        self._epoch = torch.load(self._path(hp_str + "epoch"))
        self._examples_processed = torch.load(self._path(hp_str + "examples_processed"))
        self._time_taken = torch.load(self._path(hp_str + "time_taken"))
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

        res_arr = []
        
        for i in range(5000):
            
            path = self._path(self.hp_str) + "/rkl_new_test" 
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
                
            f_name = path + "/sample_epoch_" + str(self._epoch) + "_image_" + str(i) + ".png"
            
            if os.path.isfile(f_name):
                continue

            else:
                print("------------------ Sampling " + str(i) + " out of 5000 (long) ------------------")
                sample, nll = self._model.sample(out_shape=[1024, 1])
                res_arr.append((sample, nll))
                
                if i % 100 == 0:
                
                    sample = torch.reshape(sample, [32, 32])
                    sample = sample[None, :, :]
                    sample = torch.round(127.5 * (clusters[sample.long()] + 1.0))
                    sample = sample.permute(0, 3, 1, 2)
    
                    cwd = os.getcwd()
                    dir = self._path(self.hp_str)
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    plot_images_grid(sample, f_name)
            ####################################################################################################################

            f_name = path + "/images_and_nll_samples_pixelsnail_epoch_" + str(self._epoch) + ".p"
            
            with open(f_name,'wb') as f:
             pickle.dump(res_arr,f)

    def _eval_full_model(self):
    
        
        
        def squared_euclidean_distance(a, b):
            import tensorflow as tf     
            b = tf.transpose(b)
            a2 = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)
            b2 = tf.reduce_sum(tf.square(b), axis=0, keepdims=True)
            ab = tf.matmul(a, b)
            d = a2 - 2*ab + b2
            
            return d

        def color_quantize(x, np_clusters):
            import tensorflow as tf            
            clusters = tf.Variable(np_clusters, dtype=tf.float32, trainable=False)
            x = tf.reshape(x, [-1, 3])
            d = squared_euclidean_distance(x, clusters)
            out_tf = tf.argmin(d, 1)
            out_np = out_tf.numpy()
            out_torch = torch.from_numpy(out_np)
            
            return out_torch

        def _update_eval_loader(self, epoch):

            eval_dir = self.evaldir
            path = eval_dir + eval_dir.split('/')[-2] + "_epoch_" + str(epoch)

            transform = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )

            test = datasets.ImageFolder(path, transform=transform)

            test_loader = data.DataLoader(
                test,
                batch_size=1,
                num_workers=8,
            )

            self._eval_loader = test_loader

        # Evaluate full model:
        # Load Model
        # import ipdb; ipdb.set_trace()
        dir_path = self.hp_str
        from tqdm import tqdm
        import pickle
        epoch_to_sample = self.sampling_part
        print("Evaluating From Epoch:" + str(epoch_to_sample))
        self.hp_str = dir_path + "_epoch_" + str(epoch_to_sample)
        self.load_from_checkpoint()
        total_examples, total_loss = 0, collections.defaultdict(int)
        self._sample()

        # run evaluation on test set :

#        for epoch in range(20, 26, 5):
#            eval_results_arr_test = []
#            print("sampling From Epoch:" + str(epoch))
#            _update_eval_loader(self, epoch)
#            self._model.eval()
#            for batch in tqdm(self._eval_loader):
#                batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
#                x, y = batch
#                x = color_quantize(x.numpy(), clusters_np)
#                x = x[:,None]
#                x, y = x.to('cuda'), y.to('cuda')
#                n_examples = x.shape[0]
#                total_examples += n_examples
#                for key, loss in self._eval_one_batch(x, y).items():
#                    total_loss[key] += loss * n_examples
#                eval_results_arr_test.append((x.to('cpu').numpy(), loss))
#
#            eval_path = self.evaldir
#            path = eval_path + eval_path.split('/')[-2] + "_epoch_" + str(epoch) + "_ps_vs_ps_test_eval_261021.p"
#            pickle.dump(eval_results_arr_test, open(path, "wb"))

    
    
    def interleaved_train_and_eval(self, n_epochs):
        """Trains and evaluates (after each epoch) for n_epochs."""
        if self.evalFlag:
            self._eval_full_model()

        else:
#            for i, batch in enumerate(self._train_loader):
#                batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
#                x, y = batch
#                x, y = x.to('cuda'), y.to('cuda')
#                x_np = x.cpu()
#                x_np = x_np.numpy()
#                y_np = y.cpu()
#                y_np = y_np.numpy()
                # np.save("x_single_batch_train.npy", x_np)
                # np.save("y_single_batch_train.npy", y_np)
                # break
            for epoch in range(n_epochs):

                start_time = time.time()
                print("------------------ Epoch = " + str(epoch) + " ------------------")

                # Train:
                #for i, batch in enumerate(self._train_loader):
#                for i in range(1000):
#                    print(i)
#                    #batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
#                    #x, y = batch
#                    #x, y = x.to('cuda'), y.to('cuda')
#                    self._examples_processed += x.shape[0]
#                    lrs = {
#                        f"group_{i}": param["lr"]
#                        for i, param in enumerate(self._optimizer.param_groups)
#                    }
#                    self._summary_writer.add_scalars("loss/lr", lrs, self._step)
#                    loss = self._train_one_batch(x, y)
#                    self._log_loss_dict(loss, training=True)
#                    self._time_taken += time.time() - start_time
#                    start_time = time.time()
#                    self._summary_writer.add_scalar(
#                        "speed/examples_per_sec",
#                        self._examples_processed / self._time_taken,
#                        self._step,
#                    )
#                    self._summary_writer.add_scalar(
#                        "speed/millis_per_example",
#                        self._time_taken / self._examples_processed * 1000,
#                        self._step,
#                    )
#                    self._summary_writer.add_scalar(
#                        "progress/epoch", self._epoch, self._step
#                    )
#                    self._summary_writer.add_scalar("progress/step", self._step, self._step)
#                    self._step += 1
#                    
#                self._save_checkpoint()
#                #if self._epoch %  1 == 0: #self._sample_epochs
#                    #self._sample()
#                self._sample()
#                self._epoch += 1

                # Evaluate epoch:
                print("epoch : " + str(self._epoch))
                self.load_from_checkpoint()
                self._model.eval()
                total_examples, total_loss = 0, collections.defaultdict(int)
                res_arr = []
                for batch in self._eval_loader:
#                    batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
#                    x, y = batch
#                    x, y = x.to('cuda'), y.to('cuda')
                    x = np.load("x_single_batch_train.npy")
                    y = np.load("y_single_batch_train.npy")
                    x, y = torch.from_numpy(x), torch.from_numpy(y) 
                    x, y = x.to('cuda'), y.to('cuda')
                    n_examples = x.shape[0]
                    total_examples += n_examples
                    for i in range(x.shape[0]):
                          x_temp = x[i]
                          x_temp = x_temp[None, :]
                          for key, loss in self._eval_one_batch(x_temp, y[i]).items():
                              total_loss[key] += loss * n_examples
                          temp = (x_temp.to('cpu').numpy(), loss)
                          res_arr.append(temp)
                    break
                    
                loss = {key: loss / total_examples for key, loss in total_loss.items()}
                self._log_loss_dict(loss, training=False)
                
                f_name = "/home/dsi/eyalbetzalel/pytorch-generative-v6/birds_samples" + "/" + "original_images_single_batch_" + str(self._epoch) + "_res_" + str(self._epoch) + "_nll.p"
            
                with open(f_name,'wb') as f:
                    pickle.dump(res_arr,f)
                
                self._epoch += 1

                # Sample / Save cp:


            self._summary_writer.close()
