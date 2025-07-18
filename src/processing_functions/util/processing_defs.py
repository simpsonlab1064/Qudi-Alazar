__all__ = [
    "ProcessedData",
    "LiveProcessingInterface",
    "EndProcessingInterface",
]

import numpy.typing as npt
from typing import Any, Callable, Generic, TypeVar
from abc import ABC, abstractmethod

from qudi.logic.base_alazar_logic import BaseExperimentSettings
from qudi.interface.alazar_interface import BoardInfo

T = TypeVar("T", bound="BaseExperimentSettings")
S = TypeVar("S", bound="BaseExperimentSettings")


class ProcessedData:
    """
    Class that defines the return type from processing functions.
    """

    data: list[npt.NDArray[Any]]
    labels: list[str]
    extra: dict[Any, Any]

    def __init__(
        self,
        data: list[npt.NDArray[Any]],
        labels: list[str] = [],
        extra: dict[Any, Any] | None = None,
    ):
        self.data = data
        self.labels = labels
        self.extra = extra if extra is not None else {}


class LiveProcessingInterface(Generic[T], ABC):
    @abstractmethod
    def __call__(
        self,
        data: ProcessedData,
        buf: npt.NDArray[Any],  # usually np.int_ or np.float_
        settings: T,
        buffer_index: int,
        board_index: int,
        boards: list[BoardInfo],
    ) -> ProcessedData:
        pass

    @classmethod
    def from_function(
        cls,
        func: Callable[
            [
                ProcessedData,
                npt.NDArray[Any],
                T,
                int,
                int,
                list[BoardInfo],
            ],
            ProcessedData,
        ],
    ) -> "LiveProcessingInterface[T]":
        """Create an instance of the interface by wrapping a function"""

        # Create an anonymous subclass
        class FunctionWrapper(LiveProcessingInterface[S]):
            def __init__(
                self,
                wrapped_func: Callable[
                    [
                        ProcessedData,
                        npt.NDArray[Any],
                        S,
                        int,
                        int,
                        list[BoardInfo],
                    ],
                    ProcessedData,
                ],
            ):
                self._wrapped = wrapped_func

            def __call__(
                self,
                data: ProcessedData,
                buf: npt.NDArray[Any],  # usually np.int_ or np.float_
                settings: S,
                buffer_index: int,
                board_index: int,
                boards: list[BoardInfo],
            ) -> ProcessedData:
                return self._wrapped(
                    data,
                    buf,
                    settings,
                    buffer_index,
                    board_index,
                    boards,
                )

        # Check if the function has the correct signature
        from inspect import signature

        func_sig = signature(func)
        expected_sig = signature(cls.__call__)

        # Only compare parameter names and annotations after 'self'
        func_params = list(func_sig.parameters.values())
        expected_params = list(expected_sig.parameters.values())[1:]

        if len(func_params) != len(expected_params):
            raise TypeError(
                f"Function {func.__name__} has incorrect number of arguments"
            )

        for param, exp_param in zip(func_params, expected_params):
            if param.name != exp_param.name:
                raise TypeError(
                    f"Parameter {param.name} mismatch. "
                    f"Expected: {exp_param}, Got: {param}"
                )

        return FunctionWrapper(func)


class EndProcessingInterface(Generic[T], ABC):
    @abstractmethod
    def __call__(
        self,
        data: npt.NDArray[Any],  # usually np.int_ or np.float_
        settings: T,
        boards: list[BoardInfo],
    ) -> ProcessedData:
        pass

    @classmethod
    def from_function(
        cls,
        func: Callable[
            [
                npt.NDArray[Any],
                T,
                list[BoardInfo],
            ],
            ProcessedData,
        ],
    ) -> "EndProcessingInterface[T]":
        """Create an instance of the interface by wrapping a function"""

        # Create an anonymous subclass
        class FunctionWrapper(EndProcessingInterface[S]):
            def __init__(
                self,
                wrapped_func: Callable[
                    [
                        npt.NDArray[Any],
                        S,
                        list[BoardInfo],
                    ],
                    ProcessedData,
                ],
            ):
                self._wrapped = wrapped_func

            def __call__(
                self,
                data: npt.NDArray[Any],  # usually np.int_ or np.float_
                settings: S,
                boards: list[BoardInfo],
            ) -> ProcessedData:
                return self._wrapped(
                    data,
                    settings,
                    boards,
                )

        # Check if the function has the correct signature
        from inspect import signature

        func_sig = signature(func)
        expected_sig = signature(cls.__call__)

        # Only compare parameter names and annotations after 'self'
        func_params = list(func_sig.parameters.values())
        expected_params = list(expected_sig.parameters.values())[1:]

        if len(func_params) != len(expected_params):
            raise TypeError(
                f"Function {func.__name__} has incorrect number of arguments"
            )

        for param, exp_param in zip(func_params, expected_params):
            if param.name != exp_param.name:
                raise TypeError(
                    f"Parameter {param.name} mismatch. "
                    f"Expected: {exp_param}, Got: {param}"
                )

        return FunctionWrapper(func)
