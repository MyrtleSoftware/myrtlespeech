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
        kwargs["batch_norm"] = False
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
    batch_norm = kwargs["batch_norm"]
    hidden_activation_fn = kwargs["hidden_activation_fn"]
    hidden_layers = []
    for i in range(num_hidden_layers + 1):
        # Hidden activation is eventually added only to the hidden layers
        # before the last FullyConnected layer. The same is for the batch
        # norm layers.
        if i < num_hidden_layers:
            output_features = kwargs["hidden_size"]
        else:
            output_features = kwargs["out_features"]
            hidden_activation_fn = None
            batch_norm = False

        hidden_layers.append(torch.nn.Linear(input_features, output_features))
        if batch_norm:
            hidden_layers.append(torch.nn.BatchNorm1d(output_features))
        if hidden_activation_fn is not None:
            hidden_layers.append(hidden_activation_fn)

        input_features = kwargs["hidden_size"]

    fully_connected = torch.nn.Sequential(*hidden_layers)

    if not return_kwargs:
        return fully_connected
    return fully_connected, kwargs


# Tests -----------------------------------------------------------------------


@given(fully_connected_kwargs=fully_connecteds(return_kwargs=True))
def test_fully_connected_module_structure_correct_for_valid_kwargs(
    fully_connected_kwargs: Tuple[FullyConnected, Dict]
):
    """Ensures FullyConnected.fully_connected structure is correct."""
    fully_connected, kwargs = fully_connected_kwargs

    assert isinstance(fully_connected, torch.nn.Sequential)

    # configuration of each layer in Sequential depends on whether activation
    # and batch norm are present
    act_fn_is_none = "hidden_activation_fn" not in kwargs
    batch_norm = kwargs["batch_norm"]
    hidden_size = kwargs["hidden_size"]
    input_features = kwargs["in_features"]

    num_layers = kwargs["num_hidden_layers"] + 1
    if batch_norm:
        num_layers += kwargs["num_hidden_layers"]
    if not act_fn_is_none:
        num_layers += kwargs["num_hidden_layers"]

    assert len(fully_connected) == num_layers

    for i in range(num_layers):
        # should be alternating linear/activation_fn layers if !act_fn_is_none
        # or linear/batch_norm/activation_fn if also batch_norm is True
        if (
            (batch_norm and not act_fn_is_none and i % 3 == 0)
            or (batch_norm and act_fn_is_none and i % 2 == 0)
            or (not batch_norm and not act_fn_is_none and i % 2 == 0)
            or (not batch_norm and act_fn_is_none)
        ):
            assert isinstance(fully_connected[i], torch.nn.Linear)
            assert fully_connected[i].in_features == input_features
            if i == num_layers - 1:
                assert (
                    fully_connected[i].out_features == kwargs["out_features"]
                )
            else:
                assert fully_connected[i].out_features == hidden_size

        elif (batch_norm and not act_fn_is_none and i % 3 == 1) or (
            batch_norm and act_fn_is_none and i % 2 == 1
        ):
            assert isinstance(fully_connected[i], torch.nn.BatchNorm1d)

        elif (batch_norm and not act_fn_is_none and i % 3 == 2) or (
            not batch_norm and not act_fn_is_none and i % 2 == 1
        ):
            assert isinstance(
                fully_connected[i], type(kwargs["hidden_activation_fn"])
            )

        input_features = hidden_size
