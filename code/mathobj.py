import math
import numpy as np
from abc import abstractmethod
from enum import Enum
from utilities import *


class TrigMode(Enum):
    """
    defines possible trig modes
    """
    DEGREES = True
    RADIANS = False


class Value:
    """
    bass class for all values that the calculator will interact with
    """
    def __new__(cls, *args, **kwargs):
        """
        intelligent object creation using the Value Constructor
        eg Value(np.nan) -> Undefined()
        Value(1) -> Number(1)
        Value([]) -> NPArray([])
        :param args: the value to be turned into values
        :param kwargs: key values
        """
        if len(args) != 1:
            raise TypeError("Incorrect number of arguments")
        if isinstance(args[0], np.ndarray):
            return object.__new__(NPArray)
        elif args[0] == np.nan:
            return object.__new__(Undefined)
        elif isinstance(args[0], (float, int)):
            return object.__new__(Number)
        else:
            return NotImplemented

    @abstractmethod
    def __call__(self, *args, range: int | np.ndarray = 100, **kwargs) -> 'Value':
        raise NotImplementedError


class NPArray(Value, Wrapper):
    """
    the value wrapper for numpy arrays
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, contents: np.ndarray):
        if not isinstance(contents, np.ndarray):
            raise TypeError("contents must be of type np.ndarray, not"+type(contents).__name__)
        super(NPArray, self).__init__(contents)

    def __call__(self, *args, **kwargs):
        return self

    # trig function support
    def sin(self, mode=TrigMode.DEGREES):
        assert isinstance(mode, TrigMode)
        multiplier = 1 if mode == TrigMode.RADIANS else math.pi / 180
        return NPArray(np.sin(self.get_wrapped() * multiplier))

    def cos(self, mode=TrigMode.DEGREES):
        assert isinstance(mode, TrigMode)
        multiplier = 1 if mode == TrigMode.RADIANS else math.pi / 180
        return NPArray(np.cos(self.get_wrapped() * multiplier))

    def tan(self, mode=TrigMode.DEGREES):
        assert isinstance(mode, TrigMode)
        multiplier = 1 if mode == TrigMode.RADIANS else math.pi / 180
        return NPArray(np.tan(self.get_wrapped() * multiplier))

    def arcsin(self, mode=TrigMode.DEGREES):
        assert isinstance(mode, TrigMode)
        multiplier = 1 if mode == TrigMode.RADIANS else 180 / math.pi
        return NPArray(np.arcsin(self.get_wrapped()) * multiplier)

    def arcos(self, mode=TrigMode.DEGREES):
        assert isinstance(mode, TrigMode)
        multiplier = 1 if mode == TrigMode.RADIANS else 180 / math.pi
        return NPArray(np.arccos(self.get_wrapped()) * multiplier)

    def arctan(self, mode=TrigMode.DEGREES):
        assert isinstance(mode, TrigMode)
        multiplier = 1 if mode == TrigMode.RADIANS else 180 / math.pi
        return NPArray(np.arctan(self.get_wrapped()) * multiplier)

    # miscellaneous other functions
    def log(self, *args):
        assert len(args) <= 1
        return NPArray(np.log(self.get_wrapped())/np.log(np.e if len(args) == 0 else args[0]))

    def sqrt(self):
        return NPArray(np.sqrt(self.get_wrapped()))

    # operator overloading to support +, -, *, /, **(aka ^)
    def __add__(self, other):
        if isinstance(other, Wrapper):
            other = other.get_wrapped()
            return NPArray(self._val+other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Wrapper):
            other = other.get_wrapped()
            return NPArray(other+self._val)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Wrapper):
            other = other.get_wrapped()
            return NPArray(self._val-other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Wrapper):
            other = other.get_wrapped()
            return NPArray(other-self._val)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Wrapper):
            other = other.get_wrapped()
            return NPArray(self._val*other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Wrapper):
            other = other.get_wrapped()
            return NPArray(other*self._val)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Wrapper):
            other = other.get_wrapped()
            return NPArray(self._val/other)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, Wrapper):
            other = other.get_wrapped()
            return NPArray(other/self._val)
        return NotImplemented

    def __pow__(self, power, modulo=None):
        if isinstance(power, Wrapper):
            power = power.get_wrapped()
            return NPArray(self._val**power)
        return NotImplemented

    def __rpow__(self, power, modulo=None):
        if isinstance(power, Wrapper):
            power = power.get_wrapped()
            return NPArray(power**self._val)
        return NotImplemented

    def __neg__(self):
        return NPArray(-self._val)


class Number(Value, Wrapper):
    """
    Value wrapper for numbers eg 0, 1, 3.2, ...
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, val: float):
        if not isinstance(val, (int, float)):
            try:
                super(Number, self).__init__(float(val))
                return
            except TypeError:
                raise TypeError("val must be of type int or float, not "+type(val).__name__)
        super(Number, self).__init__(float(val))

    def __float__(self):
        return self._val

    def __call__(self, *args, **kwargs):
        if isinstance(args[0], NPArray):
            return NPArray(np.linspace(self._val, self._val, args[0].get_wrapped().size))
        elif isinstance(args[0], Value):
            return self
        else:
            return NotImplemented

    # trig function support
    def sin(self, mode=TrigMode.DEGREES):
        assert isinstance(mode, TrigMode)
        multiplier = 1 if mode == TrigMode.RADIANS else math.pi/180
        return Number(math.sin(self.get_wrapped() * multiplier))

    def cos(self, mode=TrigMode.DEGREES):
        assert isinstance(mode, TrigMode)
        multiplier = 1 if mode == TrigMode.RADIANS else math.pi/180
        return Number(math.cos(self.get_wrapped() * multiplier))

    def tan(self, mode=TrigMode.DEGREES):
        assert isinstance(mode, TrigMode)
        multiplier = 1 if mode == TrigMode.RADIANS else math.pi/180
        return Number(math.tan(self.get_wrapped() * multiplier))

    def arcsin(self, mode=TrigMode.DEGREES):
        assert isinstance(mode, TrigMode)
        multiplier = 1 if mode == TrigMode.RADIANS else 180/math.pi
        if not (-1 <= self.get_wrapped() <= 1):
            return Undefined()
        return Number(math.asin(self.get_wrapped()) * multiplier)

    def arcos(self, mode=TrigMode.DEGREES):
        assert isinstance(mode, TrigMode)
        multiplier = 1 if mode == TrigMode.RADIANS else 180/math.pi
        if not (-1 <= self.get_wrapped() <= 1):
            return Undefined()
        return Number(math.asin(self.get_wrapped()) * multiplier)

    def arctan(self, mode=TrigMode.DEGREES):
        assert isinstance(mode, TrigMode)
        multiplier = 1 if mode == TrigMode.RADIANS else 180/math.pi
        if not (-1 <= self.get_wrapped() <= 1):
            return Undefined()
        return Number(math.atan(self.get_wrapped()) * multiplier)

    # miscellaneous function support
    def log(self, *args):
        assert len(args) <= 1
        if self.get_wrapped() <= 0:
            return Undefined()
        return Number(math.log(self.get_wrapped(), *map(float, args)))

    def sqrt(self):
        if self.get_wrapped() < 0:
            return Undefined()
        return Number(math.sqrt(self.get_wrapped()))

    # operator overloading to support +, -, *, /, **(aka ^)
    def __add__(self, other):
        if isinstance(other, Number):
            return Number(self._val+other._val)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Number):
            return Number(self._val-other._val)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            return Number(self._val*other._val)
        return NotImplemented

    def __truediv__(self, other):
        try:
            if isinstance(other, Number):
                return Number(self._val/other._val)
            return NotImplemented
        except ZeroDivisionError:
            return Undefined()

    def __pow__(self, power, modulo=None):
        if isinstance(power, Number):
            return Number(self._val**power._val)
        return NotImplemented

    def __neg__(self):
        return Number(-self._val)

    # for debugging & testing
    def __repr__(self):
        return "Number: " + str(self._val)


class Undefined(Value):
    """
    used to represent math errors
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __call__(self, *args, **kwargs):
        return self

    # trig function compatibility
    def sin(self, mode=TrigMode.DEGREES):
        return self

    def cos(self, mode=TrigMode.DEGREES):
        return self

    def tan(self, mode=TrigMode.DEGREES):
        return self

    def arcsin(self, mode=TrigMode.DEGREES):
        return self

    def arcos(self, mode=TrigMode.DEGREES):
        return self

    def arctan(self, mode=TrigMode.DEGREES):
        return self

    # other function compatibility
    def log(self, *args):
        assert len(args) <= 1
        return self

    def sqrt(self):
        return self

    # operator overloading to support +, -, *, /, **(aka ^)
    def __add__(self, other):
        return Undefined()

    def __radd__(self, other):
        return Undefined()

    def __sub__(self, other):
        return Undefined()

    def __rsub__(self, other):
        return Undefined()

    def __mul__(self, other):
        return Undefined()

    def __rmul__(self, other):
        return Undefined()

    def __truediv__(self, other):
        return Undefined()

    def __rtruediv__(self, other):
        return Undefined()

    def __pow__(self, power, modulo=None):
        return Undefined()

    def __rpow__(self, other):
        return Undefined()

    def __neg__(self):
        return Undefined()

    def __repr__(self):
        return "Undefined"


@test("Value")
def value_test():
    o = Number(5.2)
    a = Number(3.2)
    one = Number(1)

    print(o)
    print(o/one)
    print(float(o))
    print(o+a)
    print((o+Number(2)))

    print("weird stuff")
    print(o+Undefined())
    print(Undefined()+o)
    print("done adding undefined")
    print(o/Number(0))
    print(o/Number(0)+Undefined())

    print("printing undefined", Undefined())
    print("printing 6.2", Number(6.2))
    print("2/0", Number(2)/Number(0))
    try:
        print(Number(4)+Number(3))
        print("True")
    except TypeError:
        print("False")
    print("4+0.4+undef", Number(4)+Number(0.4)+Undefined())
    print("4^4", Number(4)**Number(4))
    print("4^Undefined", Number(4)**Undefined())
    print("Undefined^4", Undefined()**Number(4))
    pass
    print((NPArray(np.linspace(-10, 10, 100))+Undefined()))
    print(Number(4)/NPArray(np.linspace(-10, 10, 100)))


if __name__ == "__main__":
    value_test()
    quit()
