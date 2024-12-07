from mathobj import *
from controlunit import *
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
import tkinter


class Matrix(Value):
    def __new__(cls, *args, **kwargs):
        return object.__new__(Matrix)

    def __init__(self, h: int, l: int, contents, default=0):
        if not isinstance(h, int) or not isinstance(l, int):
            raise TypeError("type int expected for h and l")
        if h < 1 or l < 1:
            raise ValueError("dimensions of the matrix should be greater than 1x1")
        if len(contents) > h*l:
            raise ValueError("contents should be less than the size of the matrix")
        for _ in range(h*l-len(contents)):
            contents.append(default)
        self._dimension = (h, l)
        self._grid = [[contents[i*l+j] for j in range(l)] for i in range(h)]

    def get_dimension(self) -> tuple[int, int]:
        return self._dimension

    def __call__(self, *args, **kwargs):
        return self

    def get_matrix_of_minor(self, intersect: tuple[int, int]):
        if self._dimension[0] != self._dimension[1]:
            raise ValueError("cannot get matrix of minor for non-square matrices")
        n = self._dimension[0]
        array = []
        for i in range(n):
            for j in range(n):
                if i == intersect[0] or j == intersect[1]:
                    continue
                array.append(self._grid[i][j])
        return Matrix(n-1, n-1, array)

    def get_sub_matrix(self, top_left: tuple[int, int], bottom_right: tuple[int, int]):
        h = bottom_right[0] - top_left[0]+1
        l = bottom_right[1] - top_left[0]+1
        array = []
        for i in range(top_left[0], bottom_right[0]+1):
            for j in range(top_left[1], bottom_right[1]+1):
                array.append(self._grid[i][j])
        return Matrix(h, l, array)

    def transpose(self):
        return Matrix(*self._dimension[::-1], contents=[self._grid[j][i] for i in range(self._dimension[0]) for j in range(self._dimension[1])])

    def Det(self):
        if self._dimension[0] != self._dimension[1]:
            raise CalculationUnit.CalculationUnitExceptions("cannot get determinant of non-square matrix")
        if self._dimension[0] == 2:
            return self._grid[0][0]*self._grid[1][1]-self._grid[0][1]*self._grid[1][0]
        total = 0
        for i in range(self._dimension[0]):
            total += (1 - 2 * (i % 2)) * self._grid[0][i] * self.get_matrix_of_minor((0, i)).Det()
        return Number(total)

    def Inv(self):
        if self._dimension[0] != self._dimension[1]:
            raise CalculationUnit.CalculationUnitExceptions("cannot get determinant of non-square matrix")
        if self._dimension[0] == 2:
            return Matrix(2, 2, [self._grid[1][1], -self._grid[0][1], -self._grid[1][0], self._grid[0][0]])/self.Det()
        array = []
        for i in range(self._dimension[0]):
            for j in range(self._dimension[1]):
                array.append(
                    # swapped i and j to achieve transposition
                    (1-2*((i+j) % 2)) * self.get_matrix_of_minor((j, i)).Det()
                )
        return Matrix(*self._dimension, contents=array)/Number(self.Det())

    def __add__(self, other):
        if isinstance(other, Matrix):
            if self._dimension != other._dimension:
                raise CalculationUnit.CalculationUnitExceptions("cannot add matrices of different sizes, {}X{} and {}X{}".format(*self._dimension, *other._dimension))
            return Matrix(*self._dimension, contents=[self._grid[i][j]+other._grid[i][j] for i in range(self._dimension[0]) for j in range(self._dimension[1])])
        elif isinstance(other, Number):
            return Undefined()
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Number):
            return Undefined()
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Matrix):
            if self._dimension != other._dimension:
                raise CalculationUnit.CalculationUnitExceptions("cannot add matrices of different sizes, {}X{} and {}X{}".format(*self._dimension, *other._dimension))
            return Matrix(*self._dimension, contents=[self._grid[i][j] - other._grid[i][j] for i in range(self._dimension[0]) for j in range(self._dimension[1])])
        elif isinstance(other, Number):
            return Undefined()
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Number):
            return Undefined()
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            other = other.get_wrapped()
            return Matrix(*self._dimension, contents=[j*other for i in self._grid for j in i])
        elif isinstance(other, Matrix):
            if self._dimension[1] != other._dimension[0]:
                raise CalculationUnit.CalculationUnitExceptions("cannot multiply {}X{} matrices with {}X{} matrices".format(*self._dimension, *other._dimension))
            array = []
            for new_i in range(self._dimension[0]):
                new_row = []
                for new_j in range(other._dimension[1]):
                    total = 0
                    for t in range(self._dimension[1]):
                        total += self._grid[new_i][t]*other._grid[t][new_j]
                    new_row.append(total)
                array += new_row
            return Matrix(self._dimension[0], other._dimension[1], array)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return self.__mul__(other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Matrix):
            raise CalculationUnit.CalculationUnitExceptions("matrix division not supported, use the Inv function instead")
        if isinstance(other, Number):
            return self.__mul__(Number(1)/other)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return Undefined()
        return NotImplemented

    def __repr__(self):
        return "\n".join([str(line) for line in self._grid])


# testing interface
class CommandLineInterface(UserInterfaceInterface):
    """

    """
    class UserDoneEvent(Exception):
        """
        Custom exception raised to indicate user is done with the program
        """
        pass

    def get_custom_statements(self) -> dict:
        return {
            "quit": self.quit,
            "exit": self.quit,
            "makeMatrix": self.makeMatrix,
            "plot": self.plot,
            "settings": self.Settings,
            "help": self.help,
            "findQuadraticRoots": self.quadraticRoots,
            "changeToRad": lambda: self.set_settings({TrigMode: TrigMode.RADIANS}),
            "changeToDeg": lambda: self.set_settings({TrigMode: TrigMode.DEGREES})
        }

    def makeDefault3X3Matrix(self):
        return Matrix(3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1])

    def makeMatrix(self):
        while (height := input("input the height of the matrix (0<)")).isdigit() and int(height) == 0:
            pass
        while (width := input("input the width of the matrix (0<)")).isdigit() and int(width) == 0:
            pass
        height, width = int(height), int(width)
        default = 0
        array = ["   " for _ in range(height*width)]
        for index in range(height*width):
            for row_index in range(height):
                print("[", ",".join(map(lambda a: str(a).center(3, " "), array[row_index*width: row_index*width+width])), "]")
            while True:
                inp = input(">>>")
                if not inp:
                    inp = default
                    break
                try:
                    inp = int(inp)
                    break
                except ValueError:
                    pass
                try:
                    inp = float(inp)
                    break
                except ValueError:
                    pass
            array[index] = inp

        print(array)
        return Matrix(height, width, array)

    def __init__(self, visual=None, autostart=True):
        super(CommandLineInterface, self).__init__()
        self._prev_error = []
        self._history = []
        if visual is not None and not isinstance(visual, Visitor):
            raise TypeError("visual_display must be a visitor type, not "+type(visual).__name__)
        self.visual = visual
        if autostart:
            self.main_loop()

    def main_loop(self):
        """
        main program loop
        runs until user specifies to stop
        :return:
        """
        self.help()
        # pyramid of doom
        try:
            try:
                while True:
                    # user defined
                    inp = self.get_input_from_user()
                    # predefined
                    self.set_instruction(inp)
                    # user defined
                    if self.visual is not None:
                        # predefined
                        self.send_visitor(self.visual)
                    # user defined
                    self.return_output(result := self.get_calculated_results())
            except KeyboardInterrupt:
                self.Quit()  # quit raises UserDoneEvent
        except self.UserDoneEvent:
            pass

    def plot(self, func):
        separations = 10000
        x = np.linspace(-10, 10, separations)
        plt.plot(x, func(NPArray(x)).get_wrapped(), color="red")
        plt.show()

    def Settings(self):
        print("here are the settings you can change:")
        changed = {}
        for state, mode in self.get_settings().items():
            decision = input(f"do you wish to change {state.__name__}, it is currently in {mode.name}:")
            if not decision:
                continue
            index = "-1"
            for index, key in enumerate((possibilities := state.__members__).keys()):
                print(f"{index}) {key}")
            while not (inp := input("which mode do you want to change to?0-"+str(index))).isdigit():
                pass
            changed[state] = possibilities[list(possibilities.keys())[int(inp)]]
        self.set_settings(changed)

    def help(self):
        print("you can perform regular calculations such as:")
        print(""">>>2+3
>>>3-1
>>>0.2*3
>>>-0.25/3.1
>>>2^5
>>>(2+3)*5""")
        print("variables can be assigned using the format:")
        print("<name>=<value>")
        print("")
        print("functions can be declared as:")
        print("<name>(<parameter names>)=<expression with or without parameters>")
        print("for example: f(x)=x+2")
        print("you can call statements and functions using the following format <name>(<Operands>)")
        print("eg f(2+4)")
        print("these are all the statements you can use")
        print(list(self.get_custom_statements().keys()), sep="\n")
        print("enter \"help()\" to revisit this page")
        input("press enter to continue")

    def Quit(self):
        print("Thank you for using this calculator!")
        raise self.UserDoneEvent()

    def quit(self):
        self.Quit()

    def quadraticRoots(self, *coefficients: Value) -> None:
        """
        example statement that a user can call to calculate the roots of a quadratic equation
        :param coefficients: the three possible coefficients in
        :return:
        """
        if len(coefficients) != 3:
            raise CalculationUnit.ParameterMisMatchException(3, len(coefficients))
        previous_errors = [*self._prev_error]
        self.set_instruction("a;b;c;")
        priors = zip("abc", self.get_calculated_results())
        a, b, c = coefficients
        print(a, b, c)
        self.set_instruction(f"a={float(a)};b={float(b)};c={float(c)};")
        self.get_calculated_results()
        self.set_instruction("(-b+sqrt(b^2-4*a*c))/(2*a);(-b-sqrt(b^2-4*a*c))/(2*a);")
        # m, d = -b/2/a, self.get_functions()["sqrt"](b**2-4*a*c)/2/a
        self._history[-1] = (self._history[-1][0], self.get_calculated_results())
        for variable in priors:
            if variable[1] is None:
                continue
            self.set_instruction(variable[0] + "=" + str(float(variable[1])) + ";")
            self.get_calculated_results()
        self._prev_error = previous_errors

    def get_input_from_user(self) -> str:
        # re-inject error statement
        if len(self._history) > 0 and self._history[-1][1] == []:
            self._history[-1] = (self._history[-1][0], self._prev_error)
            self._prev_error = []
        print("-"*50)
        print(", ".join([topic.__name__ + ":" + str(state).split(".")[1] for topic, state in self.get_settings().items()]))
        print("\n"*(7-2*min(len(self._history), 3)))
        for i in range(min(len(self._history), 3)):
            print(">>>"+self._history[i-min(len(self._history), 3)][0][:-1])
            print(*(self._history[i-min(len(self._history), 3)][1]))
        inp = input(">>>")+";"
        self._history.append((inp, []))
        return inp

    def return_output(self, result: list | None):
        """
        returns and or displays the output to the user
        :param result: the result of the line
        :return: None
        """
        if result is None:  # if an error is thrown
            raise NotImplementedError
        if self._history[-1][1] not in ([], None):
            return
        lines = []
        for line in result:
            if line is None:
                continue
            if isinstance(line, Assignment):
                continue
            lines.append(line)
        self._history[-1] = (self._history[-1][0], lines)

    def alert_improper_input(self, improper_ness: str) -> None:
        self._prev_error.append(improper_ness)


if __name__ == "__main__":

    interface = CommandLineInterface(autostart=True)

    # print(list(map(lambda a: UserInterfaceInterface.__getattribute__(a), )))
