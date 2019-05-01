from .base import BaseFilter, tqdm
from torch import nn
from torch.optim import Adam, Optimizer
from torch.distributions import Normal, Independent
import torch


# TODO: Check if RNN is more suitable as compared to LSTM
# TODO: Does not work for values outside of training - need to normalize?
class SimpleRNN(nn.Module):
    def __init__(self, obsndim, hiddendim, hidden_size=10):
        """
        Basic recurrent neural network, taken from: https://gist.github.com/spro/ef26915065225df65c1187562eca7ec4
        :param obsndim: The dimension of the observation part of the model
        :type obsndim: int
        :param hiddendim: The dimension of the hidden part of the model
        :param hidden_size: int
        :param hidden_size: The size of the hidden cells (?)
        :type hidden_size: int
        """

        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self._outdim = 2 * hiddendim

        self.inp = nn.Linear(obsndim + self._outdim, self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, 1)
        self.out = nn.Linear(self.hidden_size, self._outdim)

    def step(self, inp, hidden=None):
        inp = self.inp(inp.view(1, -1)).unsqueeze(1)

        output, hidden = self.rnn(inp, hidden)
        output = self.out(output.squeeze(1))

        return output, hidden

    def forward(self, inp, hidden=None):
        output, _ = self.step(inp, hidden)

        return output[None]


class VariationalFilter(BaseFilter):
    def __init__(self, model, rnn=None, optimizer=Adam, n_samples=4):
        """
        Implements a variational filter like that of ...
        :param rnn: The recurrent neural network to parameterize the distributions
        :type rnn: Module
        :param optimizer: The optimizer to use
        :type optimizer: Optimizer
        :param n_samples: The number of samples to use
        :type n_samples: int
        """
        super().__init__(model)

        self._nn = rnn or SimpleRNN(model.obs_ndim, model.hidden_ndim)
        self._dist_old = None

        self._dist = None
        self._n_samples = (n_samples,)
        self._init_dist = (
            torch.zeros(self._model.hidden_ndim, requires_grad=True),
            torch.zeros(self._model.hidden_ndim, requires_grad=True)
        )

        self._optim = optimizer([*self._nn.parameters(), *self._init_dist])
        self._agg_loss = 0

    def _reset(self):
        self._agg_loss = 0

        return self

    def initialize(self):
        self._dist_old = self._init_dist

        # ===== Set up distribution ===== #
        if self._model.hidden_ndim < 2:
            self._dist = Normal(0., 1.)
        else:
            mean = torch.zeros(self._model.hidden_ndim)
            scale = torch.ones_like(mean)
            self._dist = Independent(Normal(mean, scale), 1)

        return self

    def _filter(self, y):
        self._optim.zero_grad()

        # ===== Get new mean and scale ===== #
        params = self._nn(torch.cat([y, *self._dist_old]))

        # TODO: Fix this
        mean, logscale = params[..., 0, 0], params[..., 0, 1]

        # ===== Sample ===== #
        xnew = mean + logscale.exp().sqrt() * self._dist.sample(self._n_samples)
        xold = self._dist_old[0] + self._dist_old[1].exp().sqrt() * self._dist.sample(self._n_samples)

        # ===== Loss ===== #
        m_ll = self._model.weight(y, xnew) + self._model.h_weight(xnew, xold)

        # TODO: Fix this
        entropy = Normal(mean, logscale.exp().sqrt()).entropy()

        loss = -(m_ll.mean(0) + entropy)
        loss.backward()

        self._agg_loss += loss

        self._optim.step()

        self._dist_old = (mean.detach(), logscale.detach())

        return mean.detach(), self._model.weight(y, mean) + self._model.h_weight(mean, self._dist_old[0])

    # TODO: Can we make batched faster?
    def fit(self, y, epochs=100):
        """
        Fits the filter to the observations, usually recommended as online learning is slow to converge.
        :param epochs: The number of epochs
        :type epochs: int
        :return: Self
        :rtype: VariationalFilter
        """

        bar = tqdm(range(int(epochs)), desc=str(self.__class__.__name__))

        for i in bar:
            self.reset().initialize()
            self.longfilter(y, bar=False)

            bar.set_description('{:s} - Avg. ELBO: {:.2f}'.format(str(self.__class__.__name__), -self._agg_loss[-1]))

        return self


