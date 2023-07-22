#
from __future__ import annotations


#
import torch
from .....models import (
    ModelHMM,
    ModelInitialGeneric,
    ModelTransitionGeneric,
    ModelEmissionCategorical,
    ModelEmissionStreamCategoricalPseudo,
)
from .....algorithms import AlgorithmGradientHMM, AlgorithmGradientStreamHMM
from .categorical import FrameworkCompareHMMCategorical


class FrameworkCompareHMMCategoricalStream(
    FrameworkCompareHMMCategorical[AlgorithmGradientHMM, AlgorithmGradientHMM],
):
    R"""
    Framework to compare two algorithms on categorical HMM dataset.
    Two algorithms are:
    1. PyTorch streaming HMM (testing);
    2. PyTorch categorical HMM (baseline).
    """

    def models(self: FrameworkCompareHMMCategoricalStream, /) -> None:
        R"""
        Prepare model(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._num_hiddens = self._dataset_tune.num_hiddens
        self._num_observations = self._dataset_tune.num_observations

        # Testing model.
        thrng = torch.Generator("cpu").manual_seed(self._seed)
        self._model_test = (
            ModelHMM(
                ModelInitialGeneric(self._num_hiddens, dint=None, dfloat=torch.float64, initeta=1.0),
                ModelTransitionGeneric(self._num_hiddens, 0.0, dint=None, dfloat=torch.float64, transeta=1.0),
                ModelEmissionStreamCategoricalPseudo(
                    ModelEmissionCategorical(
                        self._num_hiddens,
                        self._num_observations,
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
                ModelEmissionCategorical(
                    self._num_hiddens,
                    self._num_observations,
                    dint=None,
                    dfloat=torch.float64,
                    emiteta=1.0,
                ),
            )
            .reset(thrng)
            .to(self._device)
            .sgd(dict(lr=1.0))
        )

    def algorithms(self: FrameworkCompareHMMCategoricalStream, /) -> None:
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

    def communicate(self: FrameworkCompareHMMCategoricalStream, /) -> None:
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
        (emissions_test,) = self._algorithm_test.model.emission.parameters()

        #
        (initials_base,) = self._algorithm_base.model.initial.parameters()
        (transitions_base,) = self._algorithm_base.model.transition.parameters()
        (emissions_base,) = self._algorithm_base.model.emission.parameters()

        #
        initials_base.data.copy_(initials_test.data)
        transitions_base.data.copy_(transitions_test.data)
        emissions_base.data.copy_(emissions_test.data)

    def validate_loss(self: FrameworkCompareHMMCategoricalStream, /) -> None:
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

    def validate_metric(self: FrameworkCompareHMMCategoricalStream, /) -> None:
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

    def validate_parameter(self: FrameworkCompareHMMCategoricalStream, /) -> None:
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
        (emissions_test,) = self._algorithm_test.model.emission.parameters()

        #
        (initials_base,) = self._algorithm_base.model.initial.parameters()
        (transitions_base,) = self._algorithm_base.model.transition.parameters()
        (emissions_base,) = self._algorithm_base.model.emission.parameters()

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
            emissions_test.data.cpu().numpy(),
            emissions_base.data.cpu().numpy(),
        ), "Emission parameter does not match."

    def validate_final(self: FrameworkCompareHMMCategoricalStream, /) -> None:
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
