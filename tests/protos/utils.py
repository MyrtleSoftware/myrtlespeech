from typing import Dict
from typing import Iterable
from typing import Optional


def all_fields_set(
    proto, kwargs: Dict, to_ignore: Optional[Iterable[str]] = None
) -> None:
    """Crude check to ensure kwargs.keys() sets all fields for proto.

    Args:
        proto: A Python class for a protobuf message.
        kwargs: *All* kwargs to initialise class.
        to_ignore: kwarg keys to ignore when checking.

    Raises:
        :py:class:`ValueErrorr`: if all fields not populated.
    """
    expected_fields = set(proto.DESCRIPTOR.fields_by_name.keys())

    if to_ignore:
        expected_fields -= set(to_ignore)

    for oneof in proto.DESCRIPTOR.oneofs_by_name.values():
        oneof_names = set(f.name for f in oneof.fields)
        if len(oneof_names & set(kwargs)) != 1:
            raise ValueError(
                "oneof field %r not set correctly in %r, "
                "expected one of: %r, "
                "set fields: %r"
                % (
                    oneof.name,
                    proto.DESCRIPTOR.name,
                    oneof_names,
                    oneof_names & set(kwargs),
                )
            )
        expected_fields -= oneof_names

    if not (expected_fields <= set(kwargs.keys())):
        raise ValueError(
            "kwargs missing fields for %r, "
            "expected_fields (modulo oneof fields): %r, "
            "full kwargs: %r"
            % (proto.DESCRIPTOR.name, expected_fields, set(kwargs.keys()))
        )
