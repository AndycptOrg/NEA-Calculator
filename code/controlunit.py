#!/user/bin/env python

"""
Takes in the Abstract Syntax Tree as a JSON/dictionary format
and transforms into actions

semantics enforcer
ie enforces the coherence of the contents in the input(does it run)
example: "More people have been to Berlin than I have." looks correct, but isn't
example1: "factorial(e)" is not valid, although it fits the grammar
"""
import math
import warnings
import time
from abc import abstractmethod
from typing import Callable
import numpy as np


from lexer import *
from parser import *
from utilities import *
from mathobj import *

__all__ = ["UserInterfaceInterface", "CalculationUnit", "Visitor", "SubTree", "Nothing", "Assignment", "Expression"]


class Visitor:
    """
    base class for all visitors to specify all instances has visited method
    to comply with Visitor pattern
    """
    @abstractmethod
    def visited(self, visitee, *args):
        """
        method to customize response to certain objects
        :param visitee: the object visited
        :param args:
        :return:
        """
        raise NotImplementedError("visited method not yet implemented by subclass of Visitor, "+type(self).__name__)


class TwoStepVisitor(Visitor):
    """a subclass of visitor whose visit method occurs in two steps using generators"""
    @abstractmethod
    def visited(self, visitee):
        """
        :param visitee:
        args: arguments yielded later on
        :return:
        """
        super(TwoStepVisitor, self).visited(visitee)


class Visitee:
    """
    an object that is visited
    """
    @abstractmethod
    def visit(self, visitor: Visitor):
        raise NotImplementedError("visit method not yet implemented by subclass of Visitee, "+type(self).__name__)


class SubTree(Visitee):
    """
    a visitable tree that exposes its branches using the visitor pattern
    """
    def __init__(self, *branches: 'SubTree'):
        if any(map(lambda a: not isinstance(a, SubTree), branches)):
            raise TypeError("SubTree contents must be of SubTree type")
        self._branches = branches

    def visit(self, visitor: Visitor):
        if not isinstance(visitor, TwoStepVisitor):
            return visitor.visited(self, *map(lambda a: a.visit(visitor), self._branches))
        else:
            # first visit current node
            v = visitor.visited(self)
            next(v)  # start generator
            # then visit branches
            return v.send(list(map(lambda a: a.visit(visitor), self._branches)))

    def __repr__(self):
        return type(self).__name__+"\n\t"+"\n\t".join([*sum(map(lambda a: repr(a).split("\n"), self._branches), [])])


class Action(SubTree):
    """
    the "root" of the tree object
    """
    def __new__(cls, json: dict):
        if json["Type"] == "EmptyLine":
            return object.__new__(Nothing)

        unwrap_dict(json, "Action")
        typ = json["Type"]
        if typ == "Statement":
            return object.__new__(Statement)
        elif typ == "Assignment":
            return object.__new__(Assignment)
        elif typ == "Expression":
            return object.__new__(Expression)
        else:
            raise TypeError("json must be type Line, not "+typ)


class Nothing(Action):
    def __new__(cls, json: dict):
        return super(Action, cls).__new__(cls)

    # not really needed
    def __init__(self, json: dict):
        super(Nothing, self).__init__()


# TODO: remove test
@test("Nothing")
def Nothing_Action_test():
    print(o := Action({"Type": "EmptyLine", "Action": "None"}))
    print(o)
    print(type(o))


if __name__ == "__main__":
    Nothing_Action_test()


class Statement(Action):
    def __new__(cls, json: dict):
        return super(Action, cls).__new__(cls)

    def __init__(self, json: dict):
        # set the name of the statement
        self._name = json["Name"]["Name"]
        operands: list[SubTree] = []
        for operand in json["Operands"]["Operands"]:
            operands.append(Operand(operand))
        super(Statement, self).__init__(*operands)

    def get_name(self):
        return self._name

    def __repr__(self):
        head, tail = super(Statement, self).__repr__().split("\n", 1)
        head += ": "+self.get_name()
        return "\n".join([head, tail])


# TODO: remove tests
@test("Statement")
def Statement_tests():
    print(Action({'Type': 'Line', 'Action': {'Type': 'Statement', 'Name': {'Type': 'NameSpace', 'Name': 'Settings'}, 'Operands': {'Type': 'Operands', 'Operands': []}}}))
    print(Action(parse(tokenize("Setting();"), ("Setting",))[0].get_json()["Lines"][0]))

    print(Action(parse(tokenize("Settings();"), ("Settings",))[0].get_json()["Lines"][0]))


if __name__ == "__main__":
    Statement_tests()


class BinaryOperations(SubTree):
    """
    base class for all binary operations
    """
    def __init__(self, left: SubTree, right: SubTree):
        if not isinstance(left, SubTree) or not isinstance(right, SubTree):
            raise TypeError("Operands are not of type SubTree")
        super(BinaryOperations, self).__init__(left, right)


# all binary operations subclasses inherit the same constructor
# binary operations are all created in Expression initiation, so no __new__ needed
class Plus(BinaryOperations):
    pass


class Minus(BinaryOperations):
    pass


class Multiply(BinaryOperations):
    pass


class Divide(BinaryOperations):
    pass


class Exponentiation(BinaryOperations):
    pass


class Comparisons(BinaryOperations):
    """
    Base class for all comparisons
    """

    def __new__(cls, left: SubTree, right: SubTree, json: dict, *args, **kwargs):
        try:
            typ = json["ComparisonType"]
        except KeyError:
            raise TypeError("ComparisonOperator Expected, not "+json["Type"])
        if typ == ">":
            return object.__new__(GreaterThan)
        elif typ == "<":
            return object.__new__(LessThan)
        elif typ == ">=":
            return object.__new__(GreaterThanEquals)
        elif typ == "<=":
            return object.__new__(LessThanEquals)
        elif typ == "==":
            return object.__new__(Equals)
        else:
            raise TypeError("ComparisonOperator Expected, not "+json["Type"])

    def __init__(self, left: SubTree, right: SubTree, json: dict):
        super(Comparisons, self).__init__(left, right)


# no need to redeclare new since overridden new does the same if input is correct
# when incorrect json given to subClass of Comparisons, initialisation will fail, leading to Attribute Error
# ie GreaterThan({"ComparisonType": "<"})  fail
# debugger used
# cause of bug still not yet found/understood

class GreaterThan(Comparisons):
    def __new__(cls, *args, **kwargs):
        return object.__new__(GreaterThan)


class LessThan(Comparisons):
    def __new__(cls, *args, **kwargs):
        return object.__new__(LessThan)


class GreaterThanEquals(Comparisons):
    def __new__(cls, *args, **kwargs):
        return object.__new__(GreaterThanEquals)


class LessThanEquals(Comparisons):
    def __new__(cls, *args, **kwargs):
        return object.__new__(LessThanEquals)


class Equals(Comparisons):
    def __new__(cls, *args, **kwargs):
        return object.__new__(Equals)


class Operand(SubTree):
    def __new__(cls, json):
        typ = json["Operand"]["Type"]
        unwrap_dict(json, "Operand")
        if typ == "Expression":
            return object.__new__(Expression)
        elif typ == "Inequality":
            return object.__new__(Inequality)
        elif typ == "Conditional":
            return object.__new__(Conditional)
        else:
            raise TypeError("json must be type Operand, ")


class Inequality(Operand):
    def __new__(cls, json: dict):
        return object.__new__(cls)

    def __init__(self, json: dict):
        terms = list(map(Term, json["Terms"]))

        node = Comparisons(terms[0], terms[1], json["Comparisons"][0])
        if len(json["Comparisons"]) == 2:
            node = Comparisons(node, terms[2], json["Comparisons"][1])
        super(Inequality, self).__init__(node)


class Conditional(Operand):
    def __new__(cls, json: dict):
        return object.__new__(Conditional)

    def __init__(self, json: dict):
        super(Conditional, self).__init__(Inequality(json["Conditions"][0]), Inequality(json["Conditions"][1]))


# TODO: remove tests
@test("Inequality and Conditional")
def incon_tests():
    json = parse(tokenize("f(6>5);"))[0].get_json()["Lines"][0]
    # print(dictBeautify(json))
    print(Action(json))
    json = parse(tokenize("f(6>5|3<=2<1);"))[0].get_json()["Lines"][0]
    # print(dictBeautify(json))
    print(Action(json))


if __name__ == '__main__':
    incon_tests()


class Term(SubTree):
    def __new__(cls, json: dict, *args, **kwargs):
        """
        creates the correct object type when using the Term constructor
        :param json: dict being passed in
        :param args: other miscellaneous arguments
        :param kwargs: other keyword arguments
        """
        typ = "Undefined"
        if "Type" not in json or (typ := json["Type"]) not in ("Term", "NegatedTerm"):
            raise TypeError("json must be of type term, not "+typ)
        typ = json["Content"]["Type"] if json["Type"] not in ("NegatedTerm",) else json["Type"]

        # unite different implementations of Terms and BracketedTerm in parsing
        if "Content" in json:
            unwrap_dict(json, "Content")

        if typ == "Number":
            return object.__new__(Literal)
        elif typ == "NameSpace":
            return object.__new__(Variable)
        elif typ == "Function":
            return object.__new__(Function)
        elif typ == "NegatedTerm":
            return object.__new__(NegatedTerm)
        elif typ == "BracketedTerm":
            return object.__new__(BracketedTerm)
        else:
            raise TypeError("Unspecified Term Type")


class Literal(Term):
    def __new__(cls, json: dict, *args, **kwargs):
        # disallow Literal({"Function":...})
        # basically object.__new__(Literal)
        return super(Term, cls).__new__(cls)

    def __init__(self, json: dict | int | float | Value):
        """
        a container for values class
        :param json:
        """
        if not (isinstance(json, dict) and "Value" in json or isinstance(json, (int, float, Value))):
            raise TypeError("json must be of type dict with a \"Value key\", or an int or float or Value")
        if isinstance(json, dict):
            self._value = Number(json["Value"])
        elif isinstance(json, (int, float)):
            self._value = Number(json)
        else:
            self._value = json
        super(Literal, self).__init__()

    def __call__(self, *args, **kwargs):
        return self._value(*args)

    def get_value(self):
        return self._value

    def __repr__(self):
        return super(Literal, self).__repr__()[:-2]+": "+str(self._value)


class Variable(Term):
    def __new__(cls, json: dict, *args, **kwargs):
        return super(Term, cls).__new__(cls)

    def __init__(self, json: dict):
        self._name = json["Name"]
        super(Variable, self).__init__()

    def get_name(self) -> str:
        return self._name

    def __repr__(self):
        return super(Variable, self).__repr__()[:-2]+": "+self.get_name()


class Function(Term):
    def __new__(cls, json: dict, *args, **kwargs):
        return super(Term, cls).__new__(cls)

    def __init__(self, json: dict):
        self._name = json["Name"]["Name"]
        super(Function, self).__init__(*map(Operand, json["Operands"]["Operands"]))

    def get_name(self):
        return self._name

    def __repr__(self):
        head, tail = super(Function, self).__repr__().split("\n", 1)
        head += ": "+self.get_name()
        return "\n".join([head, tail])


class Expression(Action, Operand, Term):
    # delegation pattern
    def __new__(cls, json: dict):
        return object.__new__(cls)

    def __init__(self, json: dict):
        # dijkstra's shunting yard algorithm
        stack = Stack()
        operator_stack = Stack()
        operator_precedence_key = {
            "+": 0, "-": 0,
            "*": 1, "/": 1,
            "^": 2
        }
        for index, item in enumerate(json["TermsAndOperators"]):
            # every other item in list is a Term
            if index % 2 == 0:
                term = Term(item)
                stack.push(term)
                continue
            current_operator = item["OperatorType"]
            # empty operator stack
            # if precedence of the top of the operator stack >= precedence of current operator
            while not operator_stack.is_empty() and operator_precedence_key[operator_stack.peek()] >= operator_precedence_key[current_operator]:
                top = operator_stack.pop()
                right = stack.pop()
                left = stack.pop()
                stack.push({"+": Plus, "-": Minus, "*": Multiply, "/": Divide, "^": Exponentiation}[top](left, right))
            # push current operator onto operator_stack
            operator_stack.push(current_operator)
            continue  # redundant

        # empty operator_stack again
        while not operator_stack.is_empty():
            top = operator_stack.pop()
            right = stack.pop()
            left = stack.pop()
            stack.push({"+": Plus, "-": Minus, "*": Multiply, "/": Divide, "^": Exponentiation}[top](left, right))

        # assumed all expressions are valid,
        # so there should be one item on the stack remaining
        super(Expression, self).__init__(stack.pop())


class NegatedTerm(Term):
    def __new__(cls, json: dict, *args, **kwargs):
        return object.__new__(NegatedTerm)

    def __init__(self, json: dict):
        super(NegatedTerm, self).__init__(Term({"Type": "Term", "Content": json}))


class BracketedTerm(Term):
    def __new__(cls, json: dict, *args, **kwargs):
        return object.__new__(BracketedTerm)

    def __init__(self, json: dict):
        super(BracketedTerm, self).__init__(*Expression(json["expression"])._branches)


@test("Expression")
def Expression_tests():
    json = parse(tokenize("e^-(2+r)+x*3-func(31-2,2);"))[0].get_json()["Lines"][0]
    temp = Action(json)
    print(temp)
    json = parse(tokenize("m*(x+b)+c;"))[0].get_json()["Lines"][0]
    # print(dictBeautify(json))
    temp = Action(json)
    print(temp)


if __name__ == "__main__":
    Expression_tests()


class Assignment(Action):
    class AssignmentError(Exception):
        pass

    def __new__(cls, json: dict):
        return super(Action, cls).__new__(cls)

    def __init__(self, json: dict):
        assigned_type = json["Assigned"]["Assignee"]["Type"]
        self._equals_symbol = json["Equality"]["EqualityType"]

        if assigned_type == "NameSpace":
            pass
        elif assigned_type == "Function":
            assignee = json["Assigned"]["Assignee"]

            # if the number of parameters of user defined function is greater than 1
            # artificial limit number of parameters in user defined functions
            if len(operands := assignee["Operands"]["Operands"]) > 1:
                raise self.AssignmentError("defined functions must only have one or less parameters")

            # if there is one operand and that operand is not just a variable
            if len(operands) == 1 and (len(operands[0]["Operand"]["TermsAndOperators"]) != 1 or operands[0]["Operand"]["TermsAndOperators"][0]["Content"]["Type"] != "NameSpace"):
                raise self.AssignmentError("parameter(s) must be a variable")

            # collapse/compactify json
            assignee["Operands"] = assignee["Operands"]["Operands"]
            for index in range(len(assignee["Operands"])):
                assignee["Operands"][index] = assignee["Operands"][index]["Operand"]["TermsAndOperators"][0]["Content"]

        self._assignee = json["Assigned"]["Assignee"]
        expression = Expression(json["Expression"])
        super(Assignment, self).__init__(expression)

    def get_assignee(self):
        return self._assignee  # leaky reference

    def __repr__(self):
        head, tail = super(Assignment, self).__repr__().split("\n", 1)
        head += "\nAssignee:" + dict_beautify(self.get_assignee())
        return "\n".join([head, tail])


@test("Assignment")
def assignment_tests():
    print(Action(parse(tokenize("x = 2;"))[0].get_json()["Lines"][0]))
    print(Action(parse(tokenize("f() = x-4^r(6,9.8,-5);"))[0].get_json()["Lines"][0]))


if __name__ == "__main__":
    assignment_tests()


class CalculationUnit(Visitor):
    class CalculationUnitExceptions(Exception):
        """
        abstract base class for all exceptions raised from/by the Calculation Unit
        """
        pass

    class VariableNameNotFoundException(CalculationUnitExceptions, TypeError):
        def __init__(self, prompt: str):
            """
            custom Exception raised by the Calculation unit when a variable is called but not declared
            :param prompt: the name of the variable not found
            """
            super(TypeError, self).__init__(f"\"{prompt}\" has not been declared and therefore has no value and cannot be used")

    class ImmutableVariablesException(CalculationUnitExceptions):
        def __init__(self, var_name: str):
            """
            exception raised when an attempt was mode to change immutable variable/functions through instructions
            :param var_name: name of the immutable variable/function
            """
            super(CalculationUnit.CalculationUnitExceptions, self).__init__(f"\"{var_name}\" cannot be reassigned another value")

    class ParameterMisMatchException(CalculationUnitExceptions):
        def __init__(self, expected: int | str, given: int | str):
            """
            exception raised when calling a function, there is a parameter mismatch(wrong number of parameters, type of parameters)
            :param expected: what was expected
            :param given: what it was in reality
            """
            super(CalculationUnit.CalculationUnitExceptions, self).__init__(f"function takes {expected} arguments, {given} given")

    def __init__(self):

        # noinspection PyMethodParameters
        class Callable(SubTree, Wrapper):
            def __init__(self, functor):
                Wrapper.__init__(self, functor)
                SubTree.__init__(self)

        self._functor_wrapper = Callable

        # all Functors (and its subclasses) should have a corresponding CalculationUnit,
        # ie, the existence of a Functor object is dependent on the existence of a CalculationUnit
        # so hence why it is declared in the constructor of calculation unit
        # so when the reference to the associated calculation unit is required
        # the typical self is renamed to self1 to avoid collision and
        # access self from outside the scope of the method and into the scope of the calculation unit constructor
        # noinspection PyMethodParameters
        class Functor(Value):
            """
            abstract base class for all functors(functions)
            """

            def __new__(cls, *args, **kwargs):
                # redefined/overridden to avoid inheriting implementation from Value
                return object.__new__(cls)

            def __init__(self1):
                """
                set the variables associated with calling the function f(a,b) -> ["a", "b"]
                default: single variable "x"
                """
                self1._vars = self._params if self._params is not None else ["x"]

            def get_vars(self):
                """
                gets the variables required to call it
                :return:
                """
                return [var for var in self._vars]  # avoid escaping reference

            def node_version(self):
                """
                get the functor expression tree as the SubTree
                :return:
                """
                raise NotImplementedError

            def __call__(self, *args, **kwargs):
                raise NotImplementedError

            # operator overloading
            def __add__(self, other):
                first_root = self.node_version()
                if isinstance(other, Number):
                    second_root = Literal(other)
                elif isinstance(other, Functor):
                    second_root = other.node_version()
                else:
                    return
                return UserDefinedFunctor(Plus(first_root, second_root))

            def __sub__(self, other):
                first_root = self.node_version()
                if isinstance(other, Number):
                    second_root = Literal(other)
                elif isinstance(other, Functor):
                    second_root = other.node_version()
                else:
                    return
                return UserDefinedFunctor(Minus(first_root, second_root))

            def __mul__(self, other):
                first_root = self.node_version()
                if isinstance(other, Number):
                    second_root = Literal(other)
                elif isinstance(other, Functor):
                    second_root = other.node_version()
                else:
                    return
                return UserDefinedFunctor(Multiply(first_root, second_root))

            def __truediv__(self, other):
                first_root = self.node_version()
                if isinstance(other, Number):
                    second_root = Literal(other)
                elif isinstance(other, Functor):
                    second_root = other.node_version()
                else:
                    return
                return UserDefinedFunctor(Divide(first_root, second_root))

            def __pow__(self, other, modulo=None):
                first_root = self.node_version()
                if isinstance(other, Number):
                    second_root = Literal(other)
                elif isinstance(other, Functor):
                    second_root = other.node_version()
                else:
                    return
                return UserDefinedFunctor(Exponentiation(first_root, second_root))

            def __radd__(self, other):
                first_root = self.node_version()
                if isinstance(other, Number):
                    second_root = Literal(other)
                elif isinstance(other, Functor):
                    second_root = other.node_version()
                else:
                    return
                return UserDefinedFunctor(Plus(second_root, first_root))

            def __rsub__(self, other):
                first_root = self.node_version()
                if isinstance(other, Number):
                    second_root = Literal(other)
                elif isinstance(other, Functor):
                    second_root = other.node_version()
                else:
                    return
                return UserDefinedFunctor(Minus(second_root, first_root))

            def __rmul__(self, other):
                first_root = self.node_version()
                if isinstance(other, Number):
                    second_root = Literal(other)
                elif isinstance(other, Functor):
                    second_root = other.node_version()
                else:
                    return
                return UserDefinedFunctor(Multiply(second_root, first_root))

            def __rtruediv__(self, other):
                first_root = self.node_version()
                if isinstance(other, Number):
                    second_root = Literal(other)
                elif isinstance(other, Functor):
                    second_root = other.node_version()
                else:
                    return
                return UserDefinedFunctor(Divide(second_root, first_root))

            def __rpow__(self, other, modulo=None):
                first_root = self.node_version()
                if isinstance(other, Number):
                    second_root = Literal(other)
                elif isinstance(other, Functor):
                    second_root = other.node_version()
                else:
                    return
                return UserDefinedFunctor(Exponentiation(second_root, first_root))

            def __neg__(self):
                first_root = self.node_version()
                SubTree.__init__(root := object.__new__(NegatedTerm), first_root)
                return UserDefinedFunctor(root)

            def __lt__(self, other):
                first_root = self.node_version()
                if isinstance(other, Number):
                    second_root = Literal(other)
                elif isinstance(other, Functor):
                    second_root = other.node_version()
                else:
                    return
                return UserDefinedFunctor(LessThan(first_root, second_root))

            def __gt__(self, other):
                first_root = self.node_version()
                if isinstance(other, Number):
                    second_root = Literal(other)
                elif isinstance(other, Functor):
                    second_root = other.node_version()
                else:
                    return
                return UserDefinedFunctor(Divide(first_root, second_root))

            def __repr__(self):
                return repr(self.node_version())

        self.functor = Functor

        # noinspection PyMethodParameters
        class ValueDefinedFunctor(Functor):
            def __init__(self, name: str, *operands):
                assert len(operands) > 0
                super(ValueDefinedFunctor, self).__init__()
                self._name = name
                self._operands = operands

            def get_name(self):
                return self._name

            def node_version(self):
                return Callable(self)

            def __call__(self1, *args, **kwargs):
                result = []
                for operand in self1._operands:
                    if isinstance(operand, Functor):
                        result.append(self.call_func(operand, *args))
                    else:
                        result.append(operand)
                # if it is function composition
                if any(map(lambda a: isinstance(a, Functor), result)):
                    return self.value_defined_Functor(self1._name, *result)
                # check if name is a method of the value object
                obj = object()
                func = getattr(result[0], self1._name, obj)
                if func is obj:
                    raise CalculationUnit.VariableNameNotFoundException(self1._name)
                return self.call_func(func, *result[1:], name=self1._name)
        self.value_defined_Functor = ValueDefinedFunctor

        # noinspection PyMethodParameters
        class UserDefinedFunctor(Functor):
            def __init__(self, tree: SubTree):
                super(UserDefinedFunctor, self).__init__()
                if not isinstance(tree, SubTree):
                    raise TypeError("type SubTree expected, not "+type(tree).__name__)
                self._root = tree

            def node_version(self):
                return self._root

            def __call__(self1, *args, **kwargs):
                assert len(args) == len(self1.get_vars())
                self.add_function_frame(**{key: value for key, value in zip(self1.get_vars(), args)})
                # evaluate functor using calculation unit
                result = self1._root.visit(self)
                self.remove_function_frame()
                return result
        self.user_defined_functor = UserDefinedFunctor

        self._control_states = {
            TrigMode: TrigMode.RADIANS
        }

        self._params = None  # variable names to look out for when creating a function
        self._function_frame = Stack()  # call stack for identifying variables

        self._variables = {
            "Ans": Number(0),
            "x": UserDefinedFunctor(Variable({"Type": "Variable", "Name": "x"})),
            "e": Number(np.e),
            "pi": Number(np.pi)
        }

        self._IMMUTABLE_VARIABLES = {"x", "e", "pi"}  # specifies what variables cannot be changed

        self._functions = {}
        self._IMMUTABLE_FUNCTIONS = {"sqrt", "ln", "sin", "cos", "arcsin", "arcos"}  # specifies what functions cannot be user defined

    def add_function_frame(self, **kwargs):
        """
        the calculation unit is also used when resolving SubTrees in functors,
        to prevent mix-up of variables, a call stack is created to ensure variables defined outside the functor is not used when resolving the functor
        :param kwargs:
        :return:
        """
        self._function_frame.push(kwargs)

    def remove_function_frame(self):
        """
        pops function frame stack
        :return:
        """
        self._function_frame.pop()

    def begin_function_creation(self, var: list[str]):
        """
        sets flag variables to set calculation unit into function creation mode
        :param var:
        :return:
        """
        if self._params is not None:
            warnings.warn("Calculation Unit is in the process of creating a function")
            time.sleep(0)
        if not all(map(lambda a: isinstance(a, str), var)):
            raise TypeError("all variables to be passed in to var is to be type str")
        self._params = var
        self.add_function_frame(**{name: self.user_defined_functor(Variable({"Name": name})) for name in self._params})

    def end_function_creation(self):
        """
        sets flag variables to set calculation unit out of function creation mode
        :return:
        """
        if self._params is None:
            warnings.warn("Calculation Unit was not creating a function")
            time.sleep(0)
        self._params = None
        self.remove_function_frame()

    def reset_visited_trackers(self):
        """
        resets calculation unit flag variables
        :return:
        """
        if __name__ == "__main__" and (not self._function_frame.is_empty() or self._params is not None):
            print("anomaly detected", self._params, self._function_frame)
        self._params = None
        self._function_frame = Stack()

    def set_control_states(self, new_states: dict):
        """
        sets the control states of the calculation unit
        :param new_states:
        :return:
        """
        if not isinstance(new_states, dict):
            raise TypeError(f"dict type expected, got {type(new_states).__name__} instead")
        for key, value in new_states.items():
            if key not in self._control_states:
                continue
            if value not in key:
                continue
            self._control_states[key] = value

    def get_control_states(self) -> dict[Enum: Enum]:
        """
        getter for control states
        :return:
        """
        return {key: value for key, value in self._control_states.items()}  # disallow escaping reference

    def call_func(self, func, *params, name=None):
        # gives a function the proper contexts for mainly trigonometric functions
        if isinstance(func, self.value_defined_Functor):
            name = func.get_name()
        elif name is None and not isinstance(func, self.user_defined_functor):
            raise TypeError("name expected for custom functors/functions")

        trig_related_functions = ("sin", "cos", "tan", "arcsin", "arcos", "arctan")
        if name in trig_related_functions:
            return func(*params, mode=self._control_states[TrigMode])
        else:
            return func(*params)

    def visited(self, visitee, *args):
        """
        rules for each Node variant/implementation
        :param visitee: node
        :param args: results from branches using the same rules
        :return: Value
        """
        # if visiting a literal
        if isinstance(visitee, Literal):
            return visitee.get_value()
        elif isinstance(visitee, self._functor_wrapper):
            if isinstance(visitee.get_wrapped(), self.value_defined_Functor):
                return self.call_func(visitee.get_wrapped(), *self._function_frame.peek().values())
            # return visitee(*args)
        # if visiting a variable
        elif isinstance(visitee, Variable):
            if not self._function_frame.is_empty() and visitee.get_name() in self._function_frame.peek():
                return self._function_frame.peek()[visitee.get_name()]
            name = visitee.get_name()
            if name not in self._variables:
                raise self.VariableNameNotFoundException(name)
            return self._variables[name]
        # if visiting an expression or bracketed Term
        elif isinstance(visitee, (Expression, BracketedTerm, Inequality)):
            return args[0]
        elif isinstance(visitee, Plus):
            return args[0] + args[1]
        elif isinstance(visitee, Minus):
            return args[0] - args[1]
        elif isinstance(visitee, Multiply):
            return args[0] * args[1]
        elif isinstance(visitee, Divide):
            return args[0] / args[1]
        elif isinstance(visitee, Exponentiation):
            return args[0] ** args[1]
        elif isinstance(visitee, NegatedTerm):
            return -args[0]
        elif isinstance(visitee, Function):
            name = visitee.get_name()

            composition = any(map(lambda a: isinstance(a, self.functor), args))
            if composition:
                if name not in self._functions:
                    return self.value_defined_Functor(visitee.get_name(), *args)
                else:
                    return self.call_func(self._functions[name], *args)
            else:
                if len(args) == 0:
                    raise self.VariableNameNotFoundException(visitee.get_name())
                obj = object()  # used to identify if a method is present in the first argument
                func = getattr(args[0], name, obj)
                if func is not obj:  # if there is a function defined by the argument ie sin
                    return self.call_func(func, *args[1:], name=name)
                del obj
                if name not in self._functions:
                    raise self.VariableNameNotFoundException(visitee.get_name())
                else:
                    func = self._functions[name]
                    try:
                        return self.call_func(func, *args)
                    except TypeError as e:
                        if "positional argument" in str(e):  # if the incorrect number of arguments is given
                            raise CalculationUnit.ParameterMisMatchException(f"not {len(args)}", len(args))
                        else:
                            raise e
        elif isinstance(visitee, Conditional):
            return args[0] | args[1]
        elif isinstance(visitee, GreaterThan):
            return args[0] > args[1]
        elif isinstance(visitee, GreaterThanEquals):
            return args[0] >= args[1]
        elif isinstance(visitee, LessThan):
            return args[0] < args[1]
        elif isinstance(visitee, LessThanEquals):
            return args[0] <= args[1]
        elif isinstance(visitee, Equals):
            return args[0] == args[1]
        else:
            raise NotImplementedError("Calculations for " + type(visitee).__name__ + " not yet implemented")

    def set_variable(self, name: str, value):
        """
        an exposed interface for Control unit to change values of variables internally
        :param name: name of variable
        :param value: the value to set to
        :return:
        """
        if not isinstance(name, str):
            raise TypeError("str expected, not "+type(name).__name__)
        if name in self._IMMUTABLE_VARIABLES:
            raise self.ImmutableVariablesException(name)
        self._variables[name] = value

    def get_variables(self) -> dict[str: Value]:
        """
        an exposed interface for control unit to access variables declared
        :return:
        """
        return {key: value for key, value in self._variables.items()}

    def set_function(self, name: str, function):
        """
        an exposed interface for Control unit to change values of functions internally
        :param name: name of variable
        :param function: the function to set to
        :return:
        """
        if name in self._IMMUTABLE_FUNCTIONS:
            raise self.ImmutableVariablesException(name)
        self._functions[name] = function
        pass

    def get_functions(self) -> dict:
        """
        an exposed interface for Control unit to access functions declared
        :return:
        """
        return {key: value for key, value in self._functions.items()}


class ControlUnit:
    """
    an object that is associated via composition to an interface
    processes input from the interface
    proxy for
    """
    def __init__(self, interface: 'UserInterfaceInterface'):
        if not isinstance(interface, UserInterfaceInterface):
            raise TypeError("UserInterfaceInterface expected, not "+type(interface).__name__)
        # Observer pattern/ aggregation
        self._interface: UserInterfaceInterface = interface

        # the statements/commands the interface supports
        self._statements: dict[str, Callable] = self._interface.get_custom_statements()

        # the working current working lines
        self._lines: list[Action] = []

        # delegation pattern/ composition
        self._VM = CalculationUnit()

    def _interpret(self, inp: str) -> None | dict:
        """
        protected method
        tokenizes and parses input into action
        :param inp: user input in the form of a string
        :return:
        """
        try:
            tokens: list[Token] = tokenize(inp)

            parse_result = parse(tokens, self._statements)

            return parse_result[0].get_json()  # discard remainder of tokens

        except ScanError:
            self._interface.alert_improper_input("Unable to tokenize")
            return
        except ParseError:
            self._interface.alert_improper_input("Unable to parse")
            return

    def set_lines(self, inp: str) -> None:
        """
        method that the interface calls to give the control unit user input
        :param inp: user input in the form of a string
        :return:
        """
        # if an alert is raised
        # result: tokenized and parsed input
        if (result := self._interpret(inp)) is None:
            self._lines = []
            return
        self._lines: list[Action] = []

        # could have added more defensive programming
        # just assuming format is correct
        for line in result["Lines"]:
            try:
                self._lines.append(Action(line))
            except Assignment.AssignmentError:
                self._interface.alert_improper_input("assignment failed")


    def get_lines_visited(self, visual_generator: Visitor) -> list:
        """
        returns the visualization of the current working lines dictated by the visitor
        :param visual_generator: visitor containing rules that dictate how the Actions are to be visually represented
        :return: visual representations in a list
        """
        if not isinstance(visual_generator, Visitor):
            raise TypeError("Visitor expected, not "+type(visual_generator).__name__)

        return [line.visit(visual_generator) for line in self._lines]

    class ActionVisitor(TwoStepVisitor):
        """
        visitor to generate action for get_Lines_Action method
        responsible for assignment, statement, and Nothing handling
        """
        def __init__(self, vm: CalculationUnit, statements: dict[str, Callable]):
            if not isinstance(vm, CalculationUnit):
                raise TypeError("CalculationUnit expected, not "+type(vm).__name__)
            self._calc = vm
            self._statements = statements
            self._entered = False

        def visited(self, visitee):
            is_first_entry = False
            if not self._entered:
                is_first_entry = True
                self._entered = True

            if not isinstance(visitee, (Nothing, Statement, Assignment)):
                # delegation pattern
                args = yield
                val = self._calc.visited(visitee, *args)
                if is_first_entry:
                    self._calc.set_variable("Ans", val)
                yield val

            if isinstance(visitee, Nothing):
                yield  # dummy yield
                yield visitee
            elif isinstance(visitee, Statement):
                args = yield
                name = visitee.get_name()
                # not going to happen
                if name not in self._statements:
                    raise CalculationUnit.VariableNameNotFoundException(name)
                try:
                    statement_result = self._statements[name](*args)
                except TypeError as e:
                    if "positional argument" in str(e):
                        raise CalculationUnit.ParameterMisMatchException(f"not {len(args)}", len(args))
                    raise e
                if isinstance(statement_result, Value):
                    self._calc.set_variable("Ans", statement_result)
                yield statement_result

            elif isinstance(visitee, Assignment):
                typ = visitee.get_assignee()
                if typ["Type"] == "NameSpace":
                    val = yield
                    self._calc.set_variable(typ["Name"], val[0])
                elif typ["Type"] == "Function":
                    names = [variable["Name"] for variable in typ["Operands"]]
                    self._calc.begin_function_creation(names)
                    val = yield
                    self._calc.end_function_creation()
                    # if there is no variable in the expression defining the function
                    # make it a Functor
                    if not isinstance(val[0], self._calc.functor):
                        val[0] = self._calc.user_defined_functor(Literal(val[0]))
                    self._calc.set_function(typ["Name"]["Name"], val[0])
                else:
                    # TODO: investigate
                    input("wierid spot")
                    yield
                yield None
            else:
                raise TypeError("logic for " + type(visitee).__name__ + " not yet implemented")

    def get_lines_action(self) -> list:
        """
        gets and returns the current working lines Actions
        :return:
        """
        result_list = []
        for lines in self._lines:
            try:
                result = lines.visit(self.ActionVisitor(self._VM, self._statements))
                result_list.append(result)
            # relay all custom errors into alerts
            except CalculationUnit.CalculationUnitExceptions as e:
                self._interface.alert_improper_input(str(e))
                result_list.append(None)
            except NotImplementedError as e:
                self._interface.alert_improper_input(str(e)+"unsupported operation occurred while processing")
                result_list.append(None)
            finally:
                self._VM.reset_visited_trackers()
        return result_list

    def set_settings(self, parameters):
        """
        an interface for a UserInterfaceInterface to change the internal VM's setting/state
        :param parameters:
        :return:
        """
        self._VM.set_control_states(parameters)

    def get_settings(self) -> dict[Enum: Enum]:
        """
        glass door for interface to access VM control states
        :return:
        """
        return self._VM.get_control_states()

    def get_variables(self) -> dict[str, Value]:
        """
        allows interface access to variables set
        :return: dictionary of variables
        """
        return self._VM.get_variables()

    def get_functions(self) -> dict:
        """
        allows interface access to function set
        :return: dictionary of functions
        """
        return self._VM.get_functions()


class UserInterfaceInterface:
    """
    proxy for control unit
    """

    def __init__(self):
        """
        an interface that dictate what an interface must implement
        & separate user interface responsibility from controller
        """
        self.__controller = ControlUnit(self)

    def set_instruction(self, instruction: str):
        """
        sets instructions into the control unit
        :param instruction:
        :return:
        """
        if not isinstance(instruction, str):
            raise TypeError("str expected for instruction, not "+type(instruction).__name__)
        self.__controller.set_lines(instruction)

    def get_calculated_results(self) -> list:
        """
        returns results of instructions set within the control unit
        :return:
        """
        return self.__controller.get_lines_action()

    def send_visitor(self, visitor: Visitor):
        """
        allows interface to pass in a visitor to traverse the syntax tree
        :param visitor:
        :return:
        """
        if not isinstance(visitor, Visitor):
            raise TypeError("type Visitor expected, not "+type(visitor).__name__)
        return self.__controller.get_lines_visited(visitor)

    def set_settings(self, args: dict):
        """
        called to change/set settings of the virtual machine/calculation unit
        :return:
        """
        if not isinstance(args, dict):
            raise NotImplementedError("Settings not implemented for interface")
        # do user input
        self.__controller.set_settings(args)

    def get_settings(self) -> dict:
        """
        allows interface to grab/access current settings
        :return:
        """
        return self.__controller.get_settings()

    def get_variables(self) -> dict:
        """
        gets all defined variables
        :return:
        """
        return self.__controller.get_variables()

    def get_functions(self) -> dict:
        """
        gets all defined functions
        :return:
        """
        return self.__controller.get_functions()

    def get_custom_statements(self) -> dict[str, Callable]:
        """
        gets callable statements from the interface(itself)
        :return: a dictionary of the string names of the methods and the methods themselves
        """
        dict_of_methods = {}
        for attr in dir(self):
            thing = self.__getattribute__(attr)
            if type(thing).__name__ == "method" and "_" not in attr:
                dict_of_methods[attr] = thing
        return dict_of_methods

    @abstractmethod
    def alert_improper_input(self, improper_ness: str) -> None:
        """
        an alert to the user/interface why an unexpected action has occurred
        :param improper_ness: the reason why something unexpected has happened
        :return:
        """
        raise NotImplementedError


# noinspection PyTypeChecker
@test("interface")
def final_test():

    class DummyInterface(UserInterfaceInterface):
        def get_custom_statements(self) -> dict[str, Callable]:
            return {
                "settings": self.Settings,
                "setIntoDeg": self.setIntoDeg
            }

        def test(self, test_id: str, inp: str, expected_output: list, new_values: None | dict[str, Value] = None, new_functions: None | dict[str, Value] = None, new_states: None | dict=None):
            """
            tests inputs against expected outputs and variable sets
            :param test_id: the requirement being tested
            :param inp: tested input
            :param expected_output: the expected output
            :param new_values: the "new" values to be checked
            :return: None
            """
            assert isinstance(inp, str)
            assert isinstance(expected_output, list)
            assert new_values is None or isinstance(new_values, dict)
            assert new_functions is None or isinstance(new_functions, dict)
            assert new_states is None or isinstance(new_states, dict)
            passed = True
            self.set_instruction(inp)
            actual_results = self.get_calculated_results()
            if new_values is not None:
                actual_values = self.get_variables()
                for name in new_values:
                    passed &= repr(actual_values[name]) == repr(new_values[name])
            if new_functions is not None:
                actual_functions = self.get_functions()
                for name in new_functions:
                    passed &= repr(actual_functions[name]) == repr(new_functions[name])
                    if not passed:
                        print(repr(actual_functions[name]))
            if new_states is not None:
                actual_states = self.get_settings()
                for state in new_states:
                    passed &= actual_states[state] == new_states[state]

            passed &= len(actual_results) == len(expected_output)
            for actual, expected in zip(actual_results, expected_output):
                passed &= repr(actual) == repr(expected)
            print("{:<9}| {:<5}| {:<14}| {:<15}| {:<15}".format(test_id, str(passed), inp, str(expected_output), str(actual_results)))

        def Settings(self):
            print("you are in settings")
            pass

        def graph(self, value):
            print("graphing")
            print(value)
            return value

        def setIntoRad(self):
            self.set_settings({TrigMode: TrigMode.RADIANS})

        def setIntoDeg(self):
            self.set_settings({TrigMode: TrigMode.DEGREES})

        def showMode(self):
            print(self.get_settings())

        def alert_improper_input(self, improper_ness: str) -> None:
            print(improper_ness)


    control = DummyInterface()
    control.test("FR 2.1", "2+4;", [Number("6")])
    control.test("FR 2.2", "2-4;", [Number("-2")])
    control.test("FR 2.3", "2*3;", [Number("6")])
    control.test("FR 2.4", "6/2;", [Number("3")])
    control.test("FR 2.4.1", "1/0;", [Undefined()])
    control.test("FR 2.5", "2^3;", [Number("8")])
    control.test("FR 2.6", "-3;", [Number("-3")])
    control.test("FR 2.7", "(2+3)*4;", [Number("20")])
    control.test("FR 2.8.1", "a=3;", [None], new_values={"a": Number("3")})
    def dummy_init(self, value):
        self.value = value
    def dummy_repr(self):
        return self.value
    dummy = type("dummy", (object,), {"__init__": dummy_init, "__repr__": dummy_repr})
    control.test("FR 2.8.2", "f(x)=x+1;", [None], new_functions={"f": dummy("Plus\n	Variable: x\n	Literal: Number: 1.0")})
    control.set_instruction("g(x)=2*x;")
    control.get_calculated_results()
    control.test("FR 2.9", "f(g(4));", [Number("9")])
    control.test("FR 2.10", "4*3^2+5;", [Number("41")])
    control.test("FR 2.11", "sin(pi);", [Number("0")])
    control.test("FR 2.11", "tan(pi);", [Number("0")])
    control.test("FR 3.1", "setIntoDeg();", [None], new_states={TrigMode: TrigMode.DEGREES})
    control.test("FR 3.2", "sin(pi);", [Number(str(math.sin(math.pi**2/180))[:5])])
    control.test("FR 5", "2^(3+1);2^3+1;", [])
    def temp_visited(self, visitee, *args):
        if isinstance(visitee, Literal):
            return str(visitee.get_value().get_wrapped())
        elif isinstance(visitee, Plus):
            return args[0] + "+" + args[1]
        elif isinstance(visitee, Exponentiation):
            return " "*len(args[0]) + args[1] + "\n" + args[0] + " "*len(args[1])
        else:
            return args[0]
    TempVisitor = type("TempVisitor", (Visitor,), {"visited": temp_visited})
    print(*control.send_visitor(TempVisitor()), sep="\n\n")


if __name__ == '__main__':
    final_test()
