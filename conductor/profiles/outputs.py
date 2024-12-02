from pydantic import BaseModel
from typing import Any, Union, MutableMapping, MutableSequence


def recursive_model_dump(
    data: Union[BaseModel, MutableMapping, MutableSequence, Any],
) -> Any:
    """
    Recursively calls model_dump() on all BaseModel instances in a nested structure.

    Args:
        data (Union[BaseModel, MutableMapping, MutableSequence, Any]): The input data, which can be
        a BaseModel instance, a nested dictionary, a list, or any other type.

    Returns:
        Any: The structure with all BaseModel instances replaced by their model_dump() results.
    """
    if isinstance(data, BaseModel):
        # Dump the BaseModel to a dictionary
        return data.model_dump()
    elif isinstance(data, MutableMapping):
        # Process a dictionary recursively
        return {key: recursive_model_dump(value) for key, value in data.items()}
    elif isinstance(data, MutableSequence):
        # Process a list or tuple recursively
        return [recursive_model_dump(item) for item in data]
    else:
        # Return other data types unchanged
        return data
