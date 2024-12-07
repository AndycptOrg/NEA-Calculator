#!/user/bin/env python
"""
This module is responsible for the lexing of input into tokens defined by the BNF
"""

# for tests
import random
import string
from utilities import *

from abc import abstractmethod

# prevents import * from other files importing anything unnecessary
__all__ = ["Token", "tokenize", "tokenize", "ScanError"]


"""
language BNF

Scanner/Lexer Rules
<EOL>                 ::= ';' 
                        | '\n'
<OpenBracket>         ::= '('
<ClosedBracket>       ::= ')'
<Comma>               ::= ','
<ComparisonOperator>  ::= '<' 
                        | '>' 
                        | '<=' 
                        | '>=' 
                        | '=='
<Operator>            ::= '+' 
                        | '-' 
                        | '*' 
                        | '/' 
                        | '^'
<ConditionalOperator> ::= '|'
<Equality>            ::= '=' | '~'
<NameSpace>           ::= ('A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'I' | 'J' | 'K' | 'L' | 'M' | 'N' | 'O' | 'P' | 'Q' | 'R' | 'S' | 'T' | 'U' | 'V' | 'W' | 'X' | 'Y' | 'Z' | 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g' | 'h' | 'i' | 'j' | 'k' | 'l' | 'm' | 'n' | 'o' | 'p' | 'q' | 'r' | 's' | 't' | 'u' | 'v' | 'w' | 'x' | 'y' | 'z')
<NameSpace>           ::= <Namespace> <Digit> <NameSpace> | <Namespace> <Digit>
regex: [A-Za-z]+([A-Za-z0-9]+)*
<Number>              ::= <Digit> | <Digit> '.' <Digit>
regex: [0-9]+(\\.[0-9])?
<Digit>               ::= ('0' | '1' | '2' | '3' | '4' | '5' | '6'| '7' | '8' | '9')+
"""
# scanner / lexer


class Token(JSONable):
    """
    interface for all tokens generated by Lexer/ Scanner
    """
    @classmethod
    @abstractmethod
    def consume(cls, inputFeed: str) -> tuple['Token', str]:
        """
        tries to see if the beginning of the input feed fits the Token Type
        :raises NotCompatibleException: when the feed is not compatible
        :param inputFeed: string input
        :return: an instance of the Token, and the remainder of the input feed
        """
        raise NotImplementedError("Take not implemented by "+cls.__name__)

    @abstractmethod
    def get_json(self) -> dict:
        """
        gives the JSON representation of the Token
        :return:
        """
        raise NotImplementedError("JSON method not implemented by "+type(self).__name__)

    @abstractmethod
    def __eq__(self, other):
        """
        checks if the two tokens are equal
        :param other: 
        :return: 
        """
        raise NotImplementedError("== not implemented by "+type(self).__name__)


class EOL(Token):
    """
    End of Line ";", "\n"
    """
    @classmethod
    def consume(cls, inputFeed: str) -> tuple['EOL', str]:
        if len(inputFeed) > 0 and inputFeed[0] in "\n;":
            return EOL(), inputFeed[1:]
        raise NotCompatibleException

    def get_json(self) -> dict:
        return {
            "Type": "EOL"
        }

    def __eq__(self, other):
        if not isinstance(other, EOL):
            return False
        return True


class OpenBracket(Token):
    """
    Open bracket "("
    """
    @classmethod
    def consume(cls, inputFeed: str) -> tuple['OpenBracket', str]:
        if len(inputFeed) > 0 and inputFeed[0] == "(":
            return OpenBracket(), inputFeed[1:]
        raise NotCompatibleException

    def get_json(self) -> dict:
        return {
            "Type": "OpenBracket"
        }

    def __eq__(self, other):
        if not isinstance(other, OpenBracket):
            return False
        return True


class ClosedBracket(Token):
    """
    Closed bracket ")"
    """
    @classmethod
    def consume(cls, inputFeed: str) -> tuple['ClosedBracket', str]:
        if len(inputFeed) > 0 and inputFeed[0] == ")":
            return ClosedBracket(), inputFeed[1:]
        raise NotCompatibleException

    def get_json(self) -> dict:
        return {
            "Type": "ClosedBracket"
        }

    def __eq__(self, other):
        if not isinstance(other, ClosedBracket):
            return False
        return True


class Comma(Token):
    """
    Comma ","
    """
    @classmethod
    def consume(cls, inputFeed: str) -> tuple['Comma', str]:
        if len(inputFeed) > 0 and inputFeed[0] == ",":
            return Comma(), inputFeed[1:]
        raise NotCompatibleException

    def get_json(self) -> dict:
        return {
            "Type": "Comma"
        }

    def __eq__(self, other):
        return isinstance(other, Comma)


class ComparisonOperator(Token):
    """
    Comparison "<",">","==","<=",">="
    """
    def __init__(self, typeOfComparison: str):
        self._operator_type = typeOfComparison

    @classmethod
    def consume(cls, inputFeed: str) -> tuple['ComparisonOperator', str]:
        if len(inputFeed) > 1 and inputFeed[:2] in ('<=', '>=', '=='):
            return ComparisonOperator(inputFeed[:2]), inputFeed[2:]
        elif len(inputFeed) > 0 and inputFeed[0] in "<>":
            return ComparisonOperator(inputFeed[0]), inputFeed[1:]
        raise NotCompatibleException

    def get_json(self) -> dict:
        return {
            "Type": "ComparisonOperator",
            "ComparisonType": self._operator_type
        }

    # noinspection PyProtectedMember
    def __eq__(self, other):
        if not isinstance(other, ComparisonOperator):
            return False
        return self._operator_type == other._operator_type


class Operator(Token):
    """
    Operator "+","-","*","/","^"
    """
    def __init__(self, typeOfOperator: str):
        self._operator_type = typeOfOperator

    @classmethod
    def consume(cls, inputFeed: str) -> tuple['Operator', str]:
        if len(inputFeed) > 0 and inputFeed[0] in "+-*/^":
            return Operator(inputFeed[0]), inputFeed[1:]
        raise NotCompatibleException

    def get_json(self) -> dict:
        return {
            "Type": "Operator",
            "OperatorType": self._operator_type
        }

    # noinspection PyProtectedMember
    def __eq__(self, other):
        if not isinstance(other, Operator):
            return False
        return self._operator_type == other._operator_type


class ConditionalOperator(Token):
    """
    Conditional "|"
    """
    @classmethod
    def consume(cls, inputFeed: str) -> tuple['ConditionalOperator', str]:
        if len(inputFeed) > 0 and inputFeed[0] == "|":
            return ConditionalOperator(), inputFeed[1:]
        raise NotCompatibleException

    def get_json(self) -> dict:
        return {
            "Type": "ConditionalOperator"
        }

    def __eq__(self, other):
        return isinstance(other, ConditionalOperator)


class Equality(Token):
    """
    Equality "=","~"
    """
    def __init__(self, typeOfAssignment: str):
        self._assignment_type = typeOfAssignment

    @classmethod
    def consume(cls, inputFeed: str) -> tuple['Equality', str]:
        if len(inputFeed) > 0 and inputFeed[0] in "=~":
            return Equality(inputFeed[0]), inputFeed[1:]
        raise NotCompatibleException

    def get_json(self) -> dict:
        return {
            "Type": "Equality",
            "EqualityType": self._assignment_type
        }

    # noinspection PyProtectedMember
    def __eq__(self, other):
        if not isinstance(other, Equality):
            return False
        return self._assignment_type == other._assignment_type


class NameSpace(Token):
    """
    Name "..."
    """
    def __init__(self, name: str):
        self._string = name

    # noinspection PyPep8Naming
    @classmethod
    def consume(cls, inputFeed: str) -> tuple['NameSpace', str]:
        i = 0
        lenOfFeed = len(inputFeed)
        # finds the index of first non-alpha-numeric character
        while i < lenOfFeed and inputFeed[0].isalpha():
            if not inputFeed[i].isalnum():
                break
            i += 1
        # if there is no alphabet at the start, the feed is not compatible
        if i == 0:
            raise NotCompatibleException
        return NameSpace(inputFeed[:i]), inputFeed[i:]

    def get_json(self) -> dict:
        return {
            "Type": "NameSpace",
            "Name": self._string
        }

    # noinspection PyProtectedMember
    def __eq__(self, other):
        if not isinstance(other, NameSpace):
            return False
        return self._string == other._string


class Number(Token):
    """
    Number 131/12.313
    """
    def __init__(self, number: str):
        self._num = number

    # noinspection PyPep8Naming
    @classmethod
    def consume(cls, inputFeed: str) -> tuple['Number', str]:
        index = 0
        lenOfFeed = len(inputFeed)
        while index < lenOfFeed and inputFeed[index].isdigit():
            index += 1
        # if there are no numbers in the beginning of the feed
        if index == 0:
            raise NotCompatibleException
        # break off if the next character is not a decimal point
        if index == lenOfFeed or inputFeed[index] != ".":
            return Number(inputFeed[:index]), inputFeed[index:]
        t = index + 1
        while t < lenOfFeed and inputFeed[t].isdigit():
            t += 1
        # if there are no numbers after the decimal point
        if t == index + 1:
            raise NotCompatibleException
        return Number(inputFeed[:t]), inputFeed[t:]

    def get_json(self) -> dict:
        return {
            "Type": "Number",
            "Value": self._num
        }

    # noinspection PyProtectedMember
    def __eq__(self, other):
        if not isinstance(other, Number):
            return False
        return self._num == other._num


class ScanError(Exception):
    """
    error raised uniquely by tokenizer when a character sequence cannot be tokenized
    """
    def __eq__(self, other):
        if not isinstance(other, ScanError):
            return False
        return str(self) == str(other)


# noinspection PyPep8Naming
def tokenize(inputFeed: str) -> list[Token]:
    """
    scans and tokenizes into list of tokens
    :param inputFeed: string input
    :return: list of tokens
    :raises ScanError: when token stream does not cannot be tokenized
    """
    # sanitizes input
    inputFeed = inputFeed.replace(" ", "")
    # filter out unexpected characters
    # unorthodox implementation
    allowed_symbols = set("QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm~1234567890^*()-+=|;<>,./\n")
    for unexpected_symbol in set(inputFeed).difference(allowed_symbols):
        inputFeed = inputFeed.replace(unexpected_symbol, "")

    tokenList = []
    while inputFeed:
        for tokenpossibility in Token.__subclasses__():
            try:
                token, inputFeed = tokenpossibility.consume(inputFeed)
                tokenList.append(token)
                # if a token has been matched with the beginning of the feed, skip trying the rest of the tokens
                break
            except NotCompatibleException:
                # if not compatible, try the next token type
                pass
        # if all tokens have been tried, raise ScanError
        else:
            raise ScanError("the given input feed \""+inputFeed+"\" cannot be matched to any defined token")
    return tokenList


# noinspection PyPep8Naming
@test("Lexer")
def LexerTest() -> None:
    """
    tests capability of lexer, aka Scanner method
    :return: None
    """
    logFile = "../log.txt"
    # noinspection PyShadowingNames
    with open(logFile, "w") as e:
        e.write("Lexer Test Results:\n")
        e.write("""
 Test Result |   Input   |        Expected       |         Actual        |
 (Pass/fail) |           |        Outcome        |         Outcome       |
-------------+-----------+-----------------------+-----------------------+
"""[1:])

    # noinspection PyPep8Naming
    def tokenizerTest(toBeScanned: str, expectedResult: list[Token], expectedError: Exception = None) -> None:
        """
        Tests if Take method of Node obj will produce the expected result and logs result into the log.txt text file
        :param toBeScanned: typical input feed
        :param expectedResult: the node to be expected
        :param expectedError: error expected(if any)
        :return: None
        """
        # noinspection PyShadowingNames
        expected_outcome = str([*map(lambda a:type(a).__name__, expectedResult)]) if expectedError is None else (type(expectedError).__name__)
        result_text = toBeScanned.replace("\n", "\\n")
        if len(result_text) > 9:
            result_text = result_text[:6] + "..."
        result_text = f"\"{result_text}\""

        try:
            result = (outcome := tokenize(toBeScanned)) == expectedResult and expectedError is None
            if expectedError is not None:
                outcome = "Error expected: " + type(expectedError).__name__ + str(expectedError)
            outcome = str([*map(lambda a:type(a).__name__, outcome)])
        except Exception as e:
            result = e == expectedError
            if not result:
                outcome = type(e).__name__ + " " + str(e)
            else:
                outcome = type(e).__name__ + " " + str(e)
            if expectedError is None:
                result = False
        print("{:<13}|{:<11}|{:<23}|{:<23}|".format("Pass" if result else "Fail", result_text, expected_outcome, outcome), file=open(logFile, "a"))

    # testing nothing
    tokenizerTest("", [])
    # testing EOL
    tokenizerTest("\n", [EOL()])
    tokenizerTest(";", [EOL()])
    # testing OpenBracket
    tokenizerTest("(", [OpenBracket()])
    # testing ClosingBracket
    tokenizerTest(")", [ClosedBracket()])
    # testing Comma
    tokenizerTest(",", [Comma()])
    # testing ComparisonOperator
    tokenizerTest("<", [ComparisonOperator("<")])
    tokenizerTest(">", [ComparisonOperator(">")])
    tokenizerTest("<=", [ComparisonOperator("<=")])
    tokenizerTest(">=", [ComparisonOperator(">=")])
    tokenizerTest("==", [ComparisonOperator("==")])
    # testing Operator
    tokenizerTest("+", [Operator("+")])
    tokenizerTest("-", [Operator("-")])
    tokenizerTest("*", [Operator("*")])
    tokenizerTest("/", [Operator("/")])
    tokenizerTest("^", [Operator("^")])
    # testing ConditionalOperator
    tokenizerTest("|", [ConditionalOperator()])
    # testing Equality
    tokenizerTest("=", [Equality("=")])
    tokenizerTest("~", [Equality("~")])
    # testing NameSpace
    # testing first line of BNF
    tokenizerTest("a", [NameSpace("a")])
    tokenizerTest("letter", [NameSpace("letter")])
    # testing character number rule
    tokenizerTest("a1", [NameSpace("a1")])
    tokenizerTest("ae3", [NameSpace("ae3")])
    tokenizerTest("ae31", [NameSpace("ae31")])
    # testing interspersing numbers in letters rule
    possibleLetters = [*string.ascii_letters]
    possibleDigits = [*string.digits]
    for _ in range(1):
        sample = [random.choice(possibleLetters)]+[random.choice(possibleDigits)]+[random.choice(possibleLetters)]
        test = "".join(sample)
        tokenizerTest(test, [NameSpace(test)])
    # testing for ending with digit
    for _ in range(1):
        sample = [random.choice(possibleLetters)]+[random.choice(possibleDigits)]+[random.choice(possibleLetters)]+[random.choice(possibleDigits)]
        test = "".join(sample)
        tokenizerTest(test, [NameSpace(test)])

    # testing Number
    tokenizerTest("0", [Number("0")])
    tokenizerTest("123", [Number("123")])
    tokenizerTest("2.4", [Number("2.4")])
    tokenizerTest("2.44", [Number("2.44")])
    tokenizerTest("22.4", [Number("22.4")])
    tokenizerTest("62.42", [Number("62.42")])
    # testing fail/ invalid inputs
    # invalid number
    tokenizerTest(".429r4", [], ScanError("the given input feed \".429r4\" cannot be matched to any defined token"))
    tokenizerTest("84889.393.24", [], ScanError("the given input feed \".24\" cannot be matched to any defined token"))
    # letter number mash
    tokenizerTest("2944r834,e.32r.", [],
                  ScanError("the given input feed \".32r.\" cannot be matched to any defined token"))
    # testing multiple tokens in one line
    tokenizerTest("|)(7647*,<=/319.42uein~-\n2+iko2f3;ry13",
                  [ConditionalOperator(), ClosedBracket(), OpenBracket(), Number("7647"), Operator("*"), Comma(),
                   ComparisonOperator("<="), Operator("/"), Number("319.42"), NameSpace("uein"), Equality("~"),
                   Operator("-"), EOL(), Number("2"), Operator("+"), NameSpace("iko2f3"), EOL(), NameSpace("ry13")])


if __name__ == "__main__":
    LexerTest()