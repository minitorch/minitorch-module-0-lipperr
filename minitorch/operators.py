"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Return the product of x and y"""
    return x * y


def id(x: float) -> float:
    """Return unchanged input x"""
    return x


def add(x: float, y: float) -> float:
    """Return the sum of x and y"""
    return x + y


def neg(x: float) -> float:
    """Return negated x"""
    return -1.0 * x


def lt(x: float, y: float) -> bool:
    """Return True if x is less than y else False"""
    return x < y


def eq(x: float, y: float) -> float:
    """Return True if x equals y else False"""
    return x == y


def max(x: float, y: float) -> float:
    """Return maximum of x and y"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Return True if x is close to y else False"""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Return sigmoid(x)"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return relu(x) = max(0, x)"""
    return max(0.0, x)


def log(x: float) -> float:
    """Return log(x)"""
    return math.log(x)


def exp(x: float) -> float:
    """Return e^x"""
    return math.exp(x)


def inv(x: float) -> float:
    """Return the inverse of x"""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Return derivative of log in backprop"""
    return inv(x) * y


def inv_back(x: float, y: float) -> float:
    """Return derivative of inv in backprop"""
    return -inv(x * x) * y


def relu_back(x: float, y: float) -> float:
    """Return derivative of relu in backprop"""
    return int(x >= 0) * y


def map(func: Callable, arr: Iterable) -> Iterable:
    """Return an iterable object which containes outputs of func applied to elements in arr"""
    return [func(x) for x in arr]


def zipWith(func: Callable, arr1: Iterable, arr2: Iterable) -> Iterable:
    """Return an iterable object which contains outputs of func applied to corresponding values in arr1 and arr2"""
    return [func(x, y) for x, y in zip(arr1, arr2)]


def reduce(func: Callable, arr: Iterable) -> float:
    """Return the result of reduce with func"""
    x = arr[0] if len(arr) > 0 else 0
    for y in arr[1:]:
        x = func(x, y)
    return x


def negList(arr: Iterable) -> Iterable:
    """Return an iterable object with negated elements of arr"""
    return map(neg, arr)


def addLists(arr1: Iterable, arr2: Iterable) -> Iterable:
    """Return an iterable object which contains sums of corresponding values in arr1 and arr2"""
    return zipWith(add, arr1, arr2)


def sum(arr: Iterable) -> float:
    """Return the sum of all elements in iterable object "arr" """
    return reduce(add, arr)


def prod(arr: Iterable) -> float:
    """Return the product of all elements in iterable object "arr" """
    return reduce(mul, arr)
