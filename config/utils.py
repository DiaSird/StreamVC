from dataclasses import is_dataclass
from dataclasses import asdict as dataclass_as_dict
from typing import Union


def get_flattened_config_dict(config) -> dict[str, Union[int, float, str, bool]]:
    assert is_dataclass(config), "config must be a dataclass"
    config_dict = dataclass_as_dict(config)
    output_dict = {}
    values_to_unpack = list(config_dict.items())

    while values_to_unpack:
        key, value = values_to_unpack.pop()
        if is_dataclass(value):
            fields = dataclass_as_dict(value)
            for field, field_value in fields.items():
                values_to_unpack.append((f"{key}.{field}", field_value))
        elif isinstance(value, (int, float, str, bool)):
            output_dict[key] = value
        elif isinstance(value, (list, tuple)):
            for i, subvalue in enumerate(value):
                values_to_unpack.append((f"{key}[{i}]", subvalue))
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                values_to_unpack.append((f"{key}[{subkey}]", subvalue))
        else:
            pass

    return output_dict
