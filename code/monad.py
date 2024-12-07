from typing import Any

testing = __name__ == "__main__"


class Monad:
    def __init__(self, state):
        self._state = state

    @classmethod
    def just(cls, target) -> 'Monad':
        """
        wraps target
        :param target: value to be wraped
        :return:
        """
        raise NotImplementedError(f"not implemented by {cls.__name__}")

    def map(self, func) -> 'Monad':
        """
        applies func to wrapped value
        :param func:
        :return:
        """
        raise NotImplementedError(f"not implemented by {type(self).__name__}")

    def bind(self, func) -> 'Monad':
        """
        applies func and unwraps value
        :param func: function to be bound
        :return: new wrapped value
        """
        raise NotImplementedError(f"not implemented by {type(self).__name__}")

    def get_internal_implementation(self):
        """
        retrieves wrapped value
        :return: internal value
        """
        return self._state

    def __rshift__(self, function) -> 'Monad':
        """
        syntax sugar to mimic haskell bind syntax
        :param function: function to be bound
        :return: new wrapped value
        """
        return self.bind(function)


class Maybe(Monad):
    def __init__(self, target):
        super(Maybe, self).__init__(target)
        self.__target = target  # duplicate of _state of Monad

    @classmethod
    def just(cls, target):
        return Maybe(target)

    def map(self, func):
        return (Maybe(func(self.__target)) if not isinstance(self.__target, Monad) else Maybe(self.__target.map(func))) if self.__target is not None else self

    def bind(self, func):
        return Maybe(func(self.__target).__target if self.__target is not None else None)

    def __eq__(self, other):
        if not isinstance(other, Maybe):
            return False
        return self.__target == other.__target

    def __str__(self):
        return f"Maybe({self.__target})"

    def __repr__(self):
        return repr(self.__target)


def MaybeTest():
    print("\ntesting maybeMonad")
    maybe_divide = lambda num2: lambda num1: Maybe(None) if ((num2 == 0) or (num1 % num2 != 0)) else Maybe(num1 // num2)
    print(
        Maybe(20)
        .bind(maybe_divide(5))
        .bind(maybe_divide(6))
        .bind(maybe_divide(2))
    )
    print(
        Maybe(210)
        >> maybe_divide(5)
        >> maybe_divide(2)
        >> maybe_divide(7)
    )
    r = Maybe(2)
    r >>= maybe_divide(2)
    print(r)
    print("finished testing maybeMonad\n")


if testing:
    MaybeTest()


class ListMonad(Monad):
    def __init__(self, *items):
        # as None is used as the end identifier
        # it is critical to remove every instance of None when instantiating ListMonad
        # to prevent a percieved false end
        if None in items:
            items = list(items)
            items.remove(None)
        # to set _state in Monad for get_internal_implementation
        # for consistency
        super(ListMonad, self).__init__(list(items))
        if len(items) == 0:
            self._head = Maybe(None)
            self._tail = Maybe(None)
        else:
            self._head = Maybe(items[0])
            self._tail = Maybe(ListMonad(*items[1:]))

    @classmethod
    def just(cls, *target):
        return ListMonad(*target)

    def head(self) -> Any | None:
        """
        getter for the first element of the list
        :return: first element of the list or None if list is empty
        """
        match self._head:
            case Maybe(_state=None):
                return None
            case Maybe(_state=state):
                return state

    def tail(self) -> 'ListMonad | None':
        """
        getter for tail of the list, None if end of list
        [].tail() -> None
        :return: tail of the list
        """
        match self._tail:
            # case when there is no head or there is no more tail
            case Maybe(_state=None) | Maybe(_state=ListMonad(_head=None)):
                return None
            case Maybe(_state=ListMonad() as t):
                return t

    def contents(self) -> tuple[None, None] | tuple[Any, 'ListMonad']:
        """
        mimics x:xs matching in haskell
        :return: a tuple containing the head and tail of the listMonad
        """
        return self.head(), self.tail()

    def map(self, func) -> 'ListMonad':
        match len(self):
            # match empty array case
            case 0:
                return self
            # match if only one element
            case 1:
                return ListMonad(head.map(func) if isinstance((head := self.head()), Monad) else func(head))
            # if there is more than one element in the list
            case int(x) if x > 1:
                temp = ListMonad(head.map(func) if isinstance((head := self.head()), Monad) else func(head))
                if temp.head() is None:
                    return self.tail().map(func)
                temp **= self.tail().map(func)
                return temp
        raise KeyError("unknown case encountered")

    def bind(self, func) -> 'ListMonad':
        return self.flatmap(func)

    def flatmap(self, func) -> 'ListMonad':
        """
        takes in a subroutine that transforms elements in the list into ListMonads and then concatenates all the ListMonads
        :param func: function that transforms elements into ListMonads
        :return: ListMonad with func applied to elements of the list and flattens into list
        """
        match len(self):
            # match empty array case
            case 0:
                return self
            # match if only one element
            case 1:
                return func(head) if not isinstance((head := self.head()), Monad) else head.map(func)
            # when there is more than one element
            case _:
                return (func(head) if not isinstance((head := self.head()), Monad) else head.map(func)) ** self.tail().flatmap(func)

    def __pow__(self, other, modulo=None) -> 'ListMonad':
        """
        List concatenation to override python ** operator mimicing Haskell syntax
        :param other: the other ListMonad to concatenate with
        :param modulo: not supported
        :return: concatenated ListMonad
        """
        if modulo is not None:
            raise NotImplementedError("modulo not implemented by ListMonad")
        match len(self):
            # match empty array case
            case 0:
                temp = ListMonad()
                temp._head = Maybe(other.head())
                temp._tail = Maybe(other.tail())
                return temp
            # match if only one element
            case 1:
                temp = ListMonad()
                temp._head = Maybe(self.head())
                temp._tail = Maybe(other)
                return temp
            case _:
                temp = ListMonad()
                temp._head = Maybe(self.head())
                temp._tail = Maybe(self.tail() ** other)
                return temp

    def __len__(self) -> int:
        """
        calculates length of the ListMonad
        :return: length of list
        """
        match (self._head, self._tail):
            # match empty array case
            case (Maybe(_state=None), Maybe(_state=None)):
                # base case
                return 0
            # match if only one element
            case (Maybe(_state=head), Maybe(_state=ListMonad(_head=Maybe(_state=None)))):
                # base case
                return 1
            case (Maybe(_state=head), Maybe(_state=ListMonad() as tail)):
                # recursive case
                return 1 + len(tail)

    def to_list(self) -> list:
        """
        converts back into python list
        :return: list
        """
        match len(self):
            # match empty array case
            case 0:
                return []
            # match if only one element
            case 1:
                return [self.head()]
            case _:
                return [self.head()]+self.tail().to_list()

    def get_internal_implementation(self) -> list:
        """
        gets the ListMonad into list
        :return:
        """
        return self.to_list()

    def __str__(self):
        """
        get the string form of the ListMonad
        :return:
        """
        match (self._head, self._tail):
            # match empty array case
            case (Maybe(_state=None), Maybe(_state=None)):
                return "[]"
            # match if only one element
            case (Maybe(_state=head), Maybe(_state=ListMonad(_head=Maybe(_state=None)))):
                return f"[{head}]"
            case (Maybe(_state=head), Maybe(_state=ListMonad() as tail)):
                return f"[{head},{str(tail)[1:-1]}]"
            case _:
                raise KeyError("unknown case encountered")

    def __repr__(self):
        # for debugging
        return str([self._head, repr(self._tail)])


def ListMonad_test():
    """testing listMonad"""
    print("\ntesting listMonad")
    l = ListMonad(3, 5, 2, 4, 5, 2)
    # print(repr(l))
    print("6 [3,5,2,4,5,2]: expected")
    print(len(l), l)
    l = l.map(lambda a: a+3)
    print("6 [6,8,5,7,8,5]: expected")
    print(len(l), l)
    l = l.map(lambda a: a//2 if a % 2 == 0 else None)
    print("3 [3,4,4]: expected")
    print(len(l), l)
    l1 = ListMonad("T", "r", "s", "t")
    print("4 [T,r,s,t]: expected")
    print(len(l1), l1)
    l2 = l ** l1
    print("7 [3,4,4,T,r,s,t]: expected")
    print(len(l2), l2)
    l1 = l1.map(lambda a: a if a != "r" else "e")
    print("4 [T,e,s,t]: expected")
    print(len(l1), l1)
    print("7 [3,4,4,T,r,s,t]: expected")
    print(len(l2), l2)
    print("[T,e,s,t,3,4,4,T,r,s,t]: expected")
    print(l1**l2)
    print("['T', 'e', 's', 't', 3, 4, 4, 'T', 'r', 's', 't']: expected")
    print((l1 ** l2).to_list())
    l = ListMonad(24)
    split = lambda num: (lambda a: ListMonad(num, a//num) if a % num == 0 and a != num else ListMonad(a))
    split2 = split(2)
    l = l.flatmap(split2)
    print(l)
    l = l.flatmap(split(3))
    print(l)
    l = l.flatmap(split2)
    print(l)
    l = l.flatmap(split2)
    print(l)

    print("done testing listMonad\n")


if testing:
    ListMonad_test()
