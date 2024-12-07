from typing import Type, Callable, TypeVar
from abc import abstractmethod

from monad import ListMonad

__all__ = ["dict_beautify", "JSONable", "NotCompatibleException", "Wrapper", "Stack", "test", "unwrap_dict"]


class Wrapper:
    """
    unified interface for all classes that wrap a value
    specifies class has get_wrapped method that returns the wrapped value
    """
    def __init__(self, val):
        """
        initializes the _val attribute for the instance
        :param val: value being wrapped
        """
        self._val = val

    def get_wrapped(self):
        """
        returns the wrapped value
        :return: wrapped value of the instance
        """
        return self._val


class JSONable:
    """
    abstract interface that specifies all instances of subclasses will implement get_json method
    that returns a dictionary representative of itself
    """
    @abstractmethod
    def get_json(self) -> dict:
        """
        gets the instance in the form of a dictionary
        :return:
        """
        raise NotImplementedError("Not Implemented by "+type(self).__name__)


class NotCompatibleException(Exception):
    """
    raised when a match attempt has failed
    used during tokenizing and parsing
    """
    def __eq__(self, other):
        """
        used to determine if an exception is equivalent to another exception
        :param other:
        :return:
        """
        if not isinstance(other, NotCompatibleException):
            return False
        return str(self) == str(other)


# noinspection PyShadowingNames
# noinspection PyPep8Naming
# noinspection PyPep8:E741
def dict_beautify(inp: list | tuple | set | dict) -> str:
    """
    prints dictionaries in readable manner
    :param inp:
    :return: inp "JSON" in proper JSON Format
    """
    # reverses the monadic list recursively
    reverse = (lambda inp:
               ListMonad()
               if len(inp) == 0 else
               reverse(inp.tail()) ** ListMonad(inp.head())
               )

    # method that adds an indent to the front of a string
    add_indent = lambda a: "  " + a

    def deal_with_commas(commaed_lines):
        """
        removes the trailing comma in the last line in lines
        :param commaed_lines: lines with terminal comma
        :return: lines without terminal comma
        """
        # if there are no items in list
        # code smell
        if len(commaed_lines) == 0:
            return ListMonad("")
        commaed_lines = reverse(commaed_lines)
        commaed_lines = reverse(
            ListMonad(commaed_lines.head()[:-1]) ** commaed_lines.tail()
        )
        return commaed_lines

    lines = ListMonad()
    if not isinstance(inp, (dict, list, tuple, set)):
        raise TypeError("Input for dictBeautify should be a dictionary, list, tuple, or set, not "+type(inp).__name__)
    if isinstance(inp, (list, tuple, set)):
        for value in inp:
            # obtains beautified JSON format from dicts, list, sets, and tuples
            if type(value) in (dict, list, tuple, set):
                lines **= ListMonad(*(dict_beautify(value) + ",").split("\n"))
            else:
                lines **= ListMonad("\"" + str(value) + "\",")
        lines = deal_with_commas(lines)
        brackets = {list: "[]", tuple: "()", set: "{}"}
        lines = (
                # adds opening bracket
                ListMonad(brackets[type(inp)][0]) **
                # indents contents
                lines.map(add_indent) **
                # adds closing bracket
                ListMonad(brackets[type(inp)][1])
        )
        return "\n".join(lines.to_list())

    for key, value in inp.items():
        key = f"\"{key}\""
        # obtain beautified JSON if value is a dictionary, list, set, or tuple
        if type(value) in (dict, list, tuple, set):
            # define a function to add key and colon to the first line
            addToStart = (lambda item, sequence:
                          ListMonad(item + sequence.head()) ** sequence.tail()
                          )
            # obtains beautified JSON format from dicts, list, sets, and tuples and transforms into ListMonad
            lines **= addToStart(key + ": ", ListMonad(*(dict_beautify(value) + ",").split("\n")))
        else:
            lines **= ListMonad(key + ": \"" + str(value) + "\",")
    # removes trailing comma
    lines: ListMonad = deal_with_commas(lines)
    # adds {} brackets to front and end
    lines = (
            ListMonad("{") **
            # and indents all contents
            lines.map(add_indent) **
            ListMonad("}")
    )
    # transforms ListMonad into list, then concatenates each item with a newline
    return "\n".join(lines.to_list())


class Stack:
    """
    a stack, FILO data type
    """
    def __init__(self):
        self._stack = []

    def is_empty(self) -> bool:
        """
        checks if stack is empty
        :return:
        """
        return len(self._stack) == 0

    def push(self, item):
        """
        pushes item onto stack
        :param item:
        :return:
        """
        self._stack.append(item)

    def pop(self):
        """
        pops and return top item from stack
        :return: top of stack
        """
        if self.is_empty():
            raise IndexError("cannot pop from empty Stack")
        return self._stack.pop()

    def peek(self):
        """
        returns top of stack
        :return: top of stack
        """
        if self.is_empty():
            raise IndexError("cannot peek from empty stack")
        return self._stack[-1]

    def __len__(self) -> int:
        """
        returns number of items in stack
        :return: number of items in the stack
        """
        return len(self._stack)


def test(name):
    """
    test wrapper
    declares tests py printing name of test (if given) at start and end of test
    :param name: name of the test
    :return:
    """
    def actual_wrapper(func):
        """
        function that returns a function that calls the test function passed in and prints declaration before and after the test is called
        :param func: test subroutine
        :return: function that prints a declaration before and after calling the test function
        """
        def _(*args, **kwargs):
            """
            function that prints the declaration before and after the test function is called
            :param args:
            :param kwargs:
            :return: what the test function returns
            """
            t = name  # stops syntax error?
            if not isinstance(t, Callable):
                t += " "
            else:
                t = ""
            print(f"\nstart of {t}tests")
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                print(f"{t}test failed, with "+type(e).__name__+" "+str(e))
                res = None
            print(f"end of {t}tests\n")
            return res
        return _

    if not isinstance(name, Callable):
        return actual_wrapper
    return actual_wrapper(name)


# declaring generics
K = TypeVar("K")
G = TypeVar("G")


def unwrap_dict(dict_obj: dict[K, G], target_key: K):
    """
    manipulates a dictionary object in reference
    unwraps dictionary into target key's dictionary
    :param dict_obj:
    :param target_key:
    :return:
    """
    if type(dict_obj) != dict:
        raise TypeError("dict_obj must be a dictionary object, not "+type(dict_obj).__name__)
    if target_key not in dict_obj:
        raise KeyError(target_key + " not found in " + str(dict_obj))
    # save the target dictionary reference/pointer into a temporary variable
    saved_reference = dict_obj[target_key]
    if not isinstance(saved_reference, dict):
        raise TypeError("contents in key value must also by a dictionary object, not "+type(saved_reference).__name__)
    dict_obj.clear()  # .clear() also clears up *some* memory
    # redefine saved dictionary in the old dictionary reference
    # copy over key value pairs in the saved dictionary into the cleared dictionary
    for key in saved_reference:
        dict_obj[key] = saved_reference[key]


@test("unwrap_dict")
def unwrap_dict_tests():
    test1 = {"Hey": "what", "inside": {"Hey": "what"}}
    print("before:", test1, sep="\n")
    unwrap_dict(test1, "inside")
    print("after:", test1, test1 == {"Hey": "what"})


if __name__ == "__main__":
    unwrap_dict_tests()
