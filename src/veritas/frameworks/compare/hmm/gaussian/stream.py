#
from __future__ import annotations


#
import torch
from .....models import (
    ModelHMM,
    ModelInitialGeneric,
    ModelTransitionGeneric,
    ModelEmissionGaussian,
    ModelEmissionStreamGaussianPseudo,
)
from .....algorithms import AlgorithmGradientHMM, AlgorithmGradientStreamHMM
from .gaussian import FrameworkCompareHMMGaussian


class FrameworkCompareHMMGaussianStream(
    FrameworkCompareHMMGaussian[AlgorithmGradientHMM, AlgorithmGradientHMM],
):
    R"""
    Framework to compare two algorithms on Gaussian HMM dataset.
    Two algorithms are:
    1. PyTorch streaming HMM (testing);
    2. PyTorch Gaussian HMM (baseline).
    """

    def models(self: FrameworkCompareHMMGaussianStream, /) -> None:
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
                ModelEmissionStreamGaussianPseudo(
                    ModelEmissionGaussian(
                        self._num_hiddens,
                        self._num_features,
                        dint=None,
                        dfloat=torch.float64,
                        emiteta=1.0,
                    ),
                ),
            )
            .reset(thrng)
            .to(self._device)
            .sgd(dict(lr=1.0))
        )

        # Baseline model.
        thrng = torch.Generator("cpu").manual_seed(self._seed)
        self._model_base = (
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

    def algorithms(self: FrameworkCompareHMMGaussianStream, /) -> None:
        R"""
        Prepare algorithm(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._algorithm_test = AlgorithmGradientStreamHMM(self._model_test, jit=self._jit)
        self._algorithm_base = AlgorithmGradientHMM(self._model_base, jit=self._jit)

    def communicate(self: FrameworkCompareHMMGaussianStream, /) -> None:
        R"""
        Communicate between two algorithms.

        Args
        ----

        Returns
        -------
        """
        #
        (initials_test,) = self._algorithm_test.model.initial.parameters()
        (transitions_test,) = self._algorithm_test.model.transition.parameters()
        (means_test, vars_test) = self._algorithm_test.model.emission.parameters()

        #
        (initials_base,) = self._algorithm_base.model.initial.parameters()
        (transitions_base,) = self._algorithm_base.model.transition.parameters()
        (means_base, vars_base) = self._algorithm_base.model.emission.parameters()

        #
        initials_base.data.copy_(initials_test.data)
        transitions_base.data.copy_(transitions_test.data)
        means_base.data.copy_(means_test.data)
        vars_base.data.copy_(vars_test.data)

    def validate_loss(self: FrameworkCompareHMMGaussianStream, /) -> None:
        R"""
        Compare losses of every epoch.

        Args
        ----

        Returns
        -------
        """
        #
        assert self.equal(self._losses_test[:, :, 0], self._losses_base[:, :, 0]), "Loss sum does not match."
        assert self.equal(self._losses_test[:, :, 1], self._losses_base[:, :, 1]), "Loss count does not match."

    def validate_metric(self: FrameworkCompareHMMGaussianStream, /) -> None:
        R"""
        Compare metrics of every epoch.

        Args
        ----

        Returns
        -------
        """
        #
        assert self.equal(self._metrics_test[:, :, 0], self._metrics_base[:, :, 0]), "Metric sum does not match."
        assert self.equal(self._metrics_test[:, :, 1], self._metrics_base[:, :, 1]), "Metric count does not match."

    def validate_parameter(self: FrameworkCompareHMMGaussianStream, /) -> None:
        R"""
        Compare parameters of every epoch.

        Args
        ----

        Returns
        -------
        """
        #
        (initials_test,) = self._algorithm_test.model.initial.parameters()
        (transitions_test,) = self._algorithm_test.model.transition.parameters()
        (means_test, vars_test) = self._algorithm_test.model.emission.parameters()

        #
        (initials_base,) = self._algorithm_base.model.initial.parameters()
        (transitions_base,) = self._algorithm_base.model.transition.parameters()
        (means_base, vars_base) = self._algorithm_base.model.emission.parameters()

        #
        assert self.equal(
            initials_test.data.cpu().numpy(),
            initials_base.data.cpu().numpy(),
        ), "Initial distribution does not match."
        assert self.equal(
            transitions_test.data.cpu().numpy(),
            transitions_base.data.cpu().numpy(),
        ), "Transition matrix does not match."
        assert self.equal(
            means_test.data.cpu().numpy(),
            means_base.data.cpu().numpy(),
        ), "Emission mean parameter does not match."
        assert self.equal(
            vars_test.data.cpu().numpy(),
            vars_base.data.cpu().numpy(),
        ), "Emission variance parameter does not match."

    def validate_final(self: FrameworkCompareHMMGaussianStream, /) -> None:
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
                outputs_test = self._model_test.forward(inputs)
                outputs_base = self._model_test.forward(inputs)
                assert len(self._model_test.inputs(inputs)) == len(self._model_base.inputs(inputs))
                assert len(self._model_test.outputs(outputs_test)) == len(self._model_base.outputs(outputs_base))
                assert len(self._model_test.initial.inputs([])) == len(self._model_base.initial.inputs([]))
                assert len(self._model_test.initial.outputs([outputs_test[0]])) == len(
                    self._model_base.initial.outputs([outputs_base[0]])
                )
                assert len(self._model_test.transition.inputs([])) == len(self._model_base.transition.inputs([]))
                assert len(self._model_test.transition.outputs([outputs_test[1]])) == len(
                    self._model_base.transition.outputs([outputs_base[1]])
                )
                assert len(self._model_test.emission.inputs([inputs[0]])) == len(
                    self._model_base.emission.inputs([inputs[0]])
                )
                assert (
                    len(self._model_test.emission.outputs([outputs_test[2], outputs_test[3]]))
                    == len(self._model_base.emission.outputs([outputs_base[2]])) + 1
                )

        # Test algorithm intermediate outputs.
        memory_test = self._algorithm_test.details(self._algorithm_test.model, self._loader_test)
        memory_base = self._algorithm_base.details(self._algorithm_base.model, self._loader_test)
        for (i, (block_test, block_base)) in enumerate(
            zip([memory_test[0], *memory_test[2:]], memory_base),
        ):
            #
            assert self.equal(
                block_test, block_base
            ), "Algorithm intermediate memory output {:d} does not match.".format(i)
