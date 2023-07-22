#
from __future__ import annotations


#
import torch
import numpy as onp
from .....models import (
    ModelHMM,
    ModelInitialGeneric,
    ModelTransitionGeneric,
    ModelEmissionGaussian,
)
from hmmlearn.hmm import GaussianHMM
from .....algorithms import AlgorithmGradientHMM, AlgorithmConventionHMM
from .gaussian import FrameworkCompareHMMGaussian


class FrameworkCompareHMMGaussianBasic(
    FrameworkCompareHMMGaussian[AlgorithmGradientHMM, AlgorithmConventionHMM],
):
    R"""
    Framework to compare two algorithms on Gaussian HMM dataset.
    Two algorithms are:
    1. PyTorch Gaussian HMM (testing);
    2. HMMLearn Gaussian HMM (baseline).
    """

    def models(self: FrameworkCompareHMMGaussianBasic, /) -> None:
        R"""
        Prepare model(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._num_hiddens = self._dataset_tune.num_hiddens
        self._num_features = self._dataset_tune.num_features

        # Testing model.
        thrng = torch.Generator("cpu").manual_seed(self._seed)
        self._model_test = (
            ModelHMM(
                ModelInitialGeneric(self._num_hiddens, dint=None, dfloat=torch.float64, initeta=1.0),
                ModelTransitionGeneric(self._num_hiddens, 0.0, dint=None, dfloat=torch.float64, transeta=1.0),
                ModelEmissionGaussian(
                    self._num_hiddens,
                    self._num_features,
                    dint=None,
                    dfloat=torch.float64,
                    emiteta=1.0,
                ),
            )
            .reset(thrng)
            .to(self._device)
            .sgd(dict(lr=1.0))
        )

        # Baseline model.
        nprng = onp.random.RandomState(self._seed)
        self._model_base = GaussianHMM(
            n_components=self._num_hiddens,
            params="stmc",
            init_params="",
            n_iter=1,
            implementation="log",
            random_state=nprng,
            means_prior=0.0,
            means_weight=0.0,
            covars_prior=0.0,
            covars_weight=0.0,
        )
        (initials,) = self._model_test.initial.parameters()
        (transitions,) = self._model_test.transition.parameters()
        (means, covars) = self._model_test.emission.parameters()
        self._model_base.n_features = self._num_features
        self._model_base.startprob_ = onp.zeros_like(initials.data.cpu().numpy())
        self._model_base.transmat_ = onp.zeros_like(transitions.data.cpu().numpy())
        self._model_base.means_ = onp.zeros_like(means.data.cpu().numpy())
        self._model_base._covars_ = onp.ones_like(covars.data.cpu().numpy())

    def algorithms(self: FrameworkCompareHMMGaussianBasic, /) -> None:
        R"""
        Prepare algorithm(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._algorithm_test = AlgorithmGradientHMM(self._model_test, jit=self._jit)
        self._algorithm_base = AlgorithmConventionHMM(self._model_base)

    def communicate(self: FrameworkCompareHMMGaussianBasic, /) -> None:
        R"""
        Communicate between two algorithms.

        Args
        ----

        Returns
        -------
        """
        #
        (initials,) = self._algorithm_test.model.initial.parameters()
        (transitions,) = self._algorithm_test.model.transition.parameters()
        (means, covars) = self._algorithm_test.model.emission.parameters()
        onp.copyto(self._algorithm_base.model.startprob_, initials.data.cpu().numpy())
        onp.copyto(self._algorithm_base.model.transmat_, transitions.data.cpu().numpy())
        onp.copyto(self._algorithm_base.model.means_, means.data.cpu().numpy())
        onp.copyto(self._algorithm_base.model._covars_, covars.data.cpu().numpy())

    def validate_loss(self: FrameworkCompareHMMGaussianBasic, /) -> None:
        R"""
        Compare losses of every epoch.

        Args
        ----

        Returns
        -------
        """
        #
        assert self.equal(self._losses_test[0, :, 0], self._losses_base[0, :, 0]), "Loss sum does not match."
        assert self.equal(self._losses_test[0, :, 1], self._losses_base[0, :, 1]), "Loss count does not match."
        assert onp.all(onp.isnan(self._losses_base[1, :, :])), "Baseline loss should be undefined."

    def validate_metric(self: FrameworkCompareHMMGaussianBasic, /) -> None:
        R"""
        Compare metrics of every epoch.

        Args
        ----

        Returns
        -------
        """
        #
        assert self.equal(self._metrics_test[0, :, 0], self._metrics_base[0, :, 0]), "Metric sum does not match."
        assert self.equal(self._metrics_test[0, :, 1], self._metrics_base[0, :, 1]), "Metric count does not match."
        assert onp.all(onp.isnan(self._metrics_base[1, :, :])), "Baseline metric should be undefined."

    def validate_parameter(self: FrameworkCompareHMMGaussianBasic, /) -> None:
        R"""
        Compare parameters of every epoch.

        Args
        ----

        Returns
        -------
        """
        #
        (initials,) = self._algorithm_test.model.initial.parameters()
        (transitions,) = self._algorithm_test.model.transition.parameters()
        (means, covars) = self._algorithm_test.model.emission.parameters()
        assert self.equal(
            initials.data.cpu().numpy(),
            self._algorithm_base.model.startprob_,
        ), "Initial distribution does not match."
        assert self.equal(
            transitions.data.cpu().numpy(),
            self._algorithm_base.model.transmat_,
        ), "Transition matrix does not match."
        assert self.equal(
            means.data.cpu().numpy(),
            self._algorithm_base.model.means_,
        ), "Emission mean parameter does not match."
        assert self.equal(
            covars.data.cpu().numpy(),
            self._algorithm_base.model._covars_,
        ), "Emission covariance parameter does not match."

    def validate_final(self: FrameworkCompareHMMGaussianBasic, /) -> None:
        R"""
        Compare final operation(s) after tuning.

        Args
        ----

        Returns
        -------
        """
        # Test algorithm memory decoders.
        (infos, *memory) = next(iter(self._loader_test))
        with torch.no_grad():
            #
            for (begins, ends) in zip(infos[0], infos[-1]):
                #
                inputs = [block[begin:end] for (block, begin, end) in zip(memory, begins, ends)]
                outputs = self._model_test.forward(inputs)
                assert len(self._model_test.inputs(inputs)) == 2
                assert len(self._model_test.outputs(outputs)) == 3
                assert len(self._model_test.initial.inputs([])) == 0
                assert len(self._model_test.initial.outputs([outputs[0]])) == 1
                assert len(self._model_test.transition.inputs([])) == 0
                assert len(self._model_test.transition.outputs([outputs[1]])) == 1
                assert len(self._model_test.emission.inputs([inputs[0]])) == 1
                assert len(self._model_test.emission.outputs([outputs[2]])) == 1

        # Test algorithm intermediate outputs.
        for (i, (block_test, block_base)) in enumerate(
            zip(
                self._algorithm_test.details(self._algorithm_test.model, self._loader_test),
                self._algorithm_base.details(self._algorithm_base.model, self._loader_test),
            ),
        ):
            #
            assert self.equal(
                block_test, block_base
            ), "Algorithm intermediate memory output {:d} does not match.".format(i)
