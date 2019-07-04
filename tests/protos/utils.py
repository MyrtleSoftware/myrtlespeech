from typing import Dict


def all_fields_set(proto, kwargs: Dict) -> None:
    """Crude check to ensure kwargs.keys() sets all fields for proto.

    Args:
        proto: A Python class for a protobuf message.
        kwargs: *All* kwargs to initialise class.

    Raises:
        :py:class:`ValueErrorr`: if all fields not populated.
    """
    expected_fields = set(proto.DESCRIPTOR.fields_by_name.keys())

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
        kwargs

    if not (expected_fields <= set(kwargs.keys())):
        raise ValueError(
            "kwargs missing fields for %r, "
            "expected_fields (modulo oneof fields): %r, "
            "full kwargs: %r"
            % (proto.DESCRIPTOR.name, expected_fields, set(kwargs.keys()))
        )
