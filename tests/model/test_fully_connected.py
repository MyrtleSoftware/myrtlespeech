from typing import Dict
from typing import Tuple
from typing import Union

import hypothesis.strategies as st
import torch
from hypothesis import given
from myrtlespeech.model.fully_connected import FullyConnected


# Fixtures and Strategies -----------------------------------------------------


@st.composite
def fully_connecteds(
    draw, return_kwargs: bool = False
) -> Union[
    st.SearchStrategy[FullyConnected],
    st.SearchStrategy[Tuple[FullyConnected, Dict]],
]:
    """Returns a SearchStrategy for FullyConnected."""
    kwargs = {}
    kwargs["in_features"] = draw(st.integers(1, 32))
    kwargs["out_features"] = draw(st.integers(1, 32))
    kwargs["num_hidden_layers"] = draw(st.integers(0, 8))
    if kwargs["num_hidden_layers"] == 0:
        kwargs["hidden_size"] = None
        kwargs["hidden_activation_fn"] = None
    else:
        kwargs["hidden_size"] = draw(st.integers(1, 32))
        kwargs["batch_norm"] = draw(st.booleans())
        kwargs["hidden_activation_fn"] = draw(
            st.sampled_from(
                [torch.nn.ReLU(), torch.nn.Hardtanh(min_val=0.0, max_val=20.0)]
            )
        )

    num_hidden_layers = kwargs["num_hidden_layers"]
    input_features = kwargs["in_features"]
    hidden_layers = []
    for i in range(num_hidden_layers + 1):
        # Hidden activation is eventually added only to the hidden layers
        # before the last FullyConnected layer. The same is for the batch norm
        # layers.
        hidden_layers.append(
            FullyConnected(
                in_features=input_features,
                out_features=kwargs["hidden_size"]
                if i < num_hidden_layers
                else kwargs["out_features"],
                hidden_activation_fn=kwargs["hidden_activation_fn"]
                if i < num_hidden_layers
                else None,
                batch_norm=kwargs["batch_norm"]
                if i < num_hidden_layers
                else False,
            )
        )
        input_features = kwargs["hidden_size"]

    fully_connected_module = torch.nn.Sequential(*hidden_layers)

    if not return_kwargs:
        return fully_connected_module
    return fully_connected_module, kwargs


# Tests -----------------------------------------------------------------------


@given(fully_connected_kwargs=fully_connecteds(return_kwargs=True))
def test_fully_connected_module_structure_correct_for_valid_kwargs(
    fully_connected_kwargs: Tuple[FullyConnected, Dict]
):
    """Ensures FullyConnected.fully_connected structure is correct."""
    fully_connected, kwargs = fully_connected_kwargs

    assert isinstance(fully_connected, torch.nn.Sequential)

    if kwargs["num_hidden_layers"] == 0:
        assert isinstance(fully_connected[0].fully_connected, torch.nn.Linear)
        assert (
            fully_connected[0].fully_connected.in_features
            == kwargs["in_features"]
        )
        assert (
            fully_connected[0].fully_connected.out_features
            == kwargs["out_features"]
        )
        return

    assert len(fully_connected) == kwargs["num_hidden_layers"] + 1

    in_features = kwargs["in_features"]
    for idx, module in enumerate(fully_connected):
        # should be alternating linear/batch_norm/activation_fn layers
        assert isinstance(module.fully_connected, torch.nn.Linear)
        assert module.fully_connected.in_features == in_features
        if idx == len(fully_connected) - 1:
            assert (
                module.fully_connected.out_features == kwargs["out_features"]
            )
        else:
            assert module.fully_connected.out_features == kwargs["hidden_size"]
            in_features = kwargs["hidden_size"]

            if kwargs["batch_norm"] and idx < kwargs["num_hidden_layers"]:
                assert isinstance(module.batch_norm, torch.nn.BatchNorm1d)
            if idx < kwargs["num_hidden_layers"]:
                assert isinstance(
                    module.activation, type(kwargs["hidden_activation_fn"])
                )
