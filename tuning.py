import itertools
from enum import Enum
from typing import Any, Iterator, Iterable, Final


class ParameterType(Enum):
    LIST = 0
    INTEGER_RANGE = 1
    FLOAT_RANGE = 2


class ParameterRange:
    """
        We need `ParameterRange` and its iterator method ::__iter__()
         in order to easily specify the range of hyper-parameters we need
         to tune.

        In the previous assignment we had some ugly nested for loops, which
         we now want to replace with a single-level for loop, based on ranges.

        To do this, we first need a range which works for integers, floats,
         lists of booleans (enabling or disabling, sometimes a choice between
         two things), and lists of strings (for specifying different solvers,
         for example).

        Next, we will define another class which provides a "range over dict",
         as we will be specifiyng the hyper-parameter ranges and steps in the form
         of a `dict[str, ParameterRange]`.
    """

    def __init__(self,
                 values: list[Any] = None,
                 begin_inclusive: int|float = None,
                 end_inclusive: int|float = None,
                 step: int|float = None
                 ) -> None:

        #print(f"[DEBUG] values: {values} |"
        #      f" begin_inclusive: {begin_inclusive} |"
        #      f" end_inclusive: {end_inclusive} |"
        #      f" step: {step}")

        if values is not None:
            #print("[DEBUG] List")

            if isinstance(values, list):
                self._values: list[Any] = values
                self._parameter_type: ParameterType = ParameterType.LIST

            else:
                raise Exception("When specifying ::values, it must be a `list` or a `None`.")

        elif (begin_inclusive is not None and
              end_inclusive is not None and
              step is not None 
        ):
            #print("[DEBUG] Range")

            if (isinstance(begin_inclusive, int) and
                isinstance(end_inclusive, int) and
                isinstance(step, int)
            ):
                self._parameter_type: ParameterType = ParameterType.INTEGER_RANGE

            elif (isinstance(begin_inclusive, float) and
                  isinstance(end_inclusive, float) and
                  isinstance(step, float)
            ):
                self._parameter_type: ParameterType = ParameterType.FLOAT_RANGE

            else:
                raise Exception("All parameters of the range should be of the same type.")

            self._begin_inclusive: int|float = begin_inclusive
            self._end_inclusive: int|float = end_inclusive
            self._step: int|float = step

        else:
            raise Exception("Unrecognized combination of parameters.")


    def __len__(self) -> int:
        if self._parameter_type == ParameterType.LIST:
            return len(self._values)
        
        if self._parameter_type == ParameterType.INTEGER_RANGE:
            return max(0, int((self._end_inclusive - self._begin_inclusive) // self._step) + 1)

        if self._parameter_type == ParameterType.FLOAT_RANGE:
            approx_upper = (self._end_inclusive + self._step / 128.0)
            return max(0, int((approx_upper - self._begin_inclusive) // self._step) + 1)


    def __iter__(self) -> Iterator[Any]:
        # __iter__() is essentially a coroutine. An "iterator" in this context
        #  is just a loop inside a coroutine, which keeps returning/"yielding"
        #  a value on each iteration of its loop.

        if self._parameter_type == ParameterType.LIST:
            yield from self._values

        elif self._parameter_type == ParameterType.INTEGER_RANGE:
            current: int = self._begin_inclusive

            while current <= self._end_inclusive:
                yield current
                current += self._step

        elif self._parameter_type == ParameterType.FLOAT_RANGE:
            current: float = self._begin_inclusive
            approximate_upper_boundary: float = (self._end_inclusive + self._step / 128.0)

            while current <= approximate_upper_boundary:
                yield current
                current += self._step


class ParameterSpace:
    """
        This is what we use as a "range over dict". We can easily add or remove
         hyper-parameters, and iterating over all possible combinations is just
         a matter of

            for configuration in <HyperparameterSpace>:
                ...do...

        In effect, we will have a `ParameterSpace` object for each model.
    """

    def __init__(self,
                 parameter_ranges: dict[str, Iterable]
                 ) -> None:

        self._parameter_ranges: dict[str, Iterable] = parameter_ranges


    def __len__(self) -> int:
        accumulator: int = 1
        for parameter_range in self._parameter_ranges:
            accumulator *= len(parameter_range)

        return accumulator


    def __iter__(self) -> Iterable[Any]:
        keys: list[str] = list(self._parameter_ranges.keys())
        ranges: Iterable[Any] = self._parameter_ranges.values()

        # We make use of `itertools` package's ::product() method, which
        #  turns a series of Iterable[Any] into a composite iterator
        #  corresponding to the cartesian product.
        #
        # Now, our model-tuning can be a single for-loop over a dict of
        #  specified parameters.

        for combination in itertools.product(*ranges):
            yield dict(zip(keys, combination))


if __name__ == "__main__":

    def test():
        # Something we might use ourselves.
        TEST_PARAMETER_SPACE_0: ParameterSpace = ParameterSpace({
            "classic range": range(2, 5, 2),
            "integer range": ParameterRange(None, 500, 1001, 500),
            "float range": ParameterRange(None, 0.0, 1.01, 0.25),
            "single float": [1000],
            "list float": [0.01, 0.1, 1.0,],
            "single bool": [False],
            "list bool": [False, True],
        })

        # A possible edge case. The product with an empty iterable should be an empty range.
        TEST_PARAMETER_SPACE_1: ParameterSpace = ParameterSpace({
            "empty list": [],

            "classic range": range(2, 5, 2),
            "integer range": ParameterRange(None, 500, 1001, 500),
            "float range": ParameterRange(None, 0.0, 1.01, 0.25),
            "single float": [1000],
            "list float": [0.01, 0.1, 1.0,],
            "single bool": [False],
            "list bool": [False, True],
        })

        # Test 1:
        print("\n---- Test 1 ----")
        count: int = 0

        for configuration in TEST_PARAMETER_SPACE_0:
            count += 1
            print(configuration, f"Count: {count}")

        print(f"Total: {count}.")


        # Test 2:
        print("\n---- Test 2 ----")
        count: int = 0

        for configuration in TEST_PARAMETER_SPACE_1:
            count += 1
            print(configuration, f"Count: {count}")

        print(f"Total: {count}.")

        print("\n---- Finished tests ----")

    test()
