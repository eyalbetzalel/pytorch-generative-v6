"""Base classes for models."""

import torch
from torch import distributions
from torch import nn

def _default_sample_fn(logits):

    return torch.multinomial(torch.squeeze(logits),1)

class AutoregressiveModel(nn.Module):
    """The base class for Autoregressive generative models. """

    def __init__(self, sample_fn=None):
        """Initializes a new AutoregressiveModel instance.

        Args:
            sample_fn: A fn(logits)->sample which takes sufficient statistics of a
                distribution as input and returns a sample from that distribution.
                Defaults to the Bernoulli distribution.
        """
        super().__init__()
        self._sample_fn = sample_fn or _default_sample_fn

    def _get_conditioned_on(self, out_shape, conditioned_on):


        assert (
            out_shape is not None or conditioned_on is not None
        ), 'Must provided one, and only one of "out_shape" or "conditioned_on"'
        if conditioned_on is None:
            device = next(self.parameters()).device
            conditioned_on = (torch.ones(out_shape) * -1).to(device)
        else:
            conditioned_on = conditioned_on.clone()
        return conditioned_on

    # TODO(eugenhotaj): This function does not handle subpixel sampling correctly.
    def sample(self, out_shape = None, conditioned_on = None):
        """Generates new samples from the model.

        Args:
            out_shape: The expected shape of the sampled output in NCHW format.  Should
                only be provided when 'conditioned_on=None'.
            conditioned_on: A batch of partial samples to condition the generation on.
                Only dimensions with values < 0 will be sampled while dimensions with
                values >= 0 will be left unchanged. If 'None', an unconditional sample
                will be generated.
        """
        with torch.no_grad():
            conditioned_on = self._get_conditioned_on(out_shape, conditioned_on)

            h, w = conditioned_on.shape
            n=1
            c=1
            conditioned_on = conditioned_on.long()


            for row in range(h):
                for col in range(w):


                    # ipdb > conditioned_on.shape
                    # torch.Size([1, 1, 1024, 1])
                    out = self.forward(conditioned_on)
                    out = out[:,:,:,None]
                    out = out[:, :, row, col]
                    import ipdb; ipdb.set_trace()
                    out = self._sample_fn(torch.exp(out)).view(n, c)
                    conditioned_on[row, col] = torch.where(
                        conditioned_on[row, col] < 0,
                        out,
                        conditioned_on[row, col],
                    )

            return conditioned_on
