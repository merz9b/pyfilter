from .base import BatchAlgorithm, SequentialAlgorithm
import torch
from torch import optim
import tqdm
from ..proposals.linearized import eps
from .varapprox import MeanField, BaseApproximation, ParameterApproximation


class VariationalBayes(BatchAlgorithm):
    def __init__(self, model, num_samples=4, approx=MeanField(), optimizer=optim.Adam, maxiters=30e3, optkwargs=None):
        """
        Implements Variational Bayes.
        :param model: The model
        :type model: pyfilter.timeseries.StateSpaceModel
        :param num_samples: The number of samples
        :type num_samples: int
        :param approx: The variational approximation to use for the latent space
        :type approx: BaseApproximation
        :param optimizer: The optimizer
        :type optimizer: optim.Optimizer
        :param maxiters: The maximum number of iterations
        :type maxiters: int|float
        :param optkwargs: Any optimizer specific kwargs
        :type optkwargs: dict
        """

        super().__init__(None)
        self._model = model
        self._numsamples = num_samples

        self._approximation = approx
        self._p_approx = None   # type: ParameterApproximation

        self._optimizer = optimizer
        self._maxiters = int(maxiters)
        self.optkwargs = optkwargs or dict()

        self._runavg = 0
        self._decay = 0.975

    @property
    def approximation(self):
        """
        Returns the resulting variational approximation
        :rtype: BaseApproximation
        """

        return self._approximation

    def loss(self, y):
        """
        The loss function, i.e. ELBO
        :rtype: torch.Tensor
        """

        # ===== Sample states ===== #
        transformed = self._approximation.sample(self._numsamples)

        # TODO: Clean this up and make it work for matrices
        # ===== Sample parameters ===== #
        params = self._p_approx.sample(self._numsamples)
        for i, p in enumerate(self._model.flat_theta_dists):
            p.t_values = params[..., i]

        # ===== Helpers ===== #
        x_t = transformed[:, 1:]
        x_tm1 = transformed[:, :-1]

        # ===== Loss function ===== #
        logl = (self._model.weight(y, x_t) + self._model.h_weight(x_t, x_tm1)).sum(1).mean(0)
        entropy = self._approximation.entropy() + self._p_approx.entropy()

        return -(logl + self._model.p_prior(transformed=True).mean() + entropy)

    def _fit(self, y):
        # ===== Initialize the state approximation ===== #
        self._approximation.initialize(y, self._model.hidden_ndim)

        # ===== Setup the parameter approximation ===== #
        self._p_approx = ParameterApproximation().initialize(self._model.flat_theta_dists)

        # ===== Define the optimizer ===== #
        parameters = [*self._approximation.get_parameters(), *self._p_approx.get_parameters()]
        optimizer = self._optimizer(parameters, **self.optkwargs)

        elbo_old = -torch.tensor(float('inf'))
        elbo = -elbo_old

        it = 0
        bar = tqdm.tqdm(total=self._maxiters)
        while (elbo - elbo_old).abs() > eps and it < self._maxiters:
            elbo_old = elbo

            # ===== Perform optimization ===== #
            optimizer.zero_grad()
            elbo = self.loss(y)
            elbo.backward()
            optimizer.step()

            it += 1
            bar.update(1)
            self._runavg = self._runavg * self._decay - elbo[-1] * (1 - self._decay)
            bar.set_description('{:s} - Avg. ELBO: {:.2f}'.format(str(self), self._runavg))

        return self


class SequentialVariationalBayes(SequentialAlgorithm):
    def __init__(self, filter_, num_samples=4, optimizer=optim.Adam, optkwargs=None):
        """
        Variational Bayes using SMC.
        :param num_samples: The number of samples
        :type num_samples: int
        :param optimizer: The optimizer
        :type optimizer: optim.Optimizer
        :param optkwargs: Any optimizer specific kwargs
        :type optkwargs: dict
        """
        super().__init__(filter_)
        self._numsamples = num_samples

        self._p_approx = None   # type: ParameterApproximation

        self._optimizer = optimizer
        self._init_optimizer = None
        self.optkwargs = optkwargs or dict()

    def _update_params(self):
        params = self._p_approx.sample(self._numsamples)
        for i, p in enumerate(self._filter.ssm.flat_theta_dists):
            p.t_values = params[..., i]

        return self

    def initialize(self):
        self._p_approx = ParameterApproximation().initialize(self._filter.ssm.flat_theta_dists)
        self._init_optimizer = self._optimizer(self._p_approx.get_parameters(), **self.optkwargs)

        self._update_params()

        self._filter.set_nparallel(self._numsamples).initialize()

        return self

    def _update(self, y):
        self._init_optimizer.zero_grad()

        self._filter.filter(y)
        (-(self._filter.s_ll[-1].mean() + self._p_approx.entropy())).backward()

        self._init_optimizer.step()
        self._update_params()

        return self


