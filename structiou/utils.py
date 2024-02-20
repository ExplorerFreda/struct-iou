'''Utility functions for the structured span aligner.'''
from typing import Tuple


def eps_eq(x: float, y: float, eps: float = 1e-6) -> bool:
    '''Check whether two values are the same under a given epsilon.

    Args:
        x: The first value.
        y: The second value.
        eps: The epsilon to use.

    Returns:
        True if the values are the same under the given epsilon, False otherwise.

    Examples:
        >>> print(eps_eq(1.0, 1.0))
        True

        >>> print(eps_eq(1.0, 1.0 + 1e-7))
        True

        >>> print(eps_eq(1.0, 1.0 + 1e-5))
        False
    '''
    return abs(x - y) < eps


def substitute_substr(string: str, old: str, new: str) -> str:
    '''Substitutes a pattern in a string with a new string: numbers are appended to the new string in case there are multiple occurrences of the pattern.

    Args:
        string: The string to substitute the pattern in.
        old: The pattern to substitute.
        new: The string to substitute the pattern with.

    Examples:
        >>> print(substitute_substr('a b c b', 'b', 'd'))
        a d_0 c d_1
    '''
    index = 0
    new_word_index = 0
    while index < len(string):
        if string[index:index+len(old)] == old:
            string = string[:index] + (f'{new}_{new_word_index}' if new is not None else f'{new_word_index}') + string[index+len(old):]
            new_word_index += 1
        index += 1
    return string


def span_iou(span1: Tuple[float, float], span2: Tuple[float, float], eps: float = 1e-6) -> float:
    '''Calculates the intersection over union ratio of two spans.

    Args:
        span1: The first span.
        span2: The second span.
        eps: The epsilon to use.

    Returns:
        The intersection over union ratio of the two spans.
    '''
    span_intersection = lambda span1, span2: (max(span1[0], span2[0]), min(span1[1], span2[1]))
    span_union = lambda span1, span2: (min(span1[0], span2[0]), max(span1[1], span2[1]))

    intersection_span = span_intersection(span1, span2)
    intersection = intersection_span[1] - intersection_span[0]
    union_span = span_union(span1, span2)
    union = union_span[1] - union_span[0]
    if intersection < eps or union < eps:
        return 0
    else:
        return intersection / union


def normalize_parentheses_string(string: str) -> str:
    '''Normalizes a string with parentheses by adding spaces around the parentheses and removing redundant spaces.

    Args:
        string: The string to normalize.

    Returns:
        The normalized string.

    Examples:
        >>> print(normalize_parentheses_string('((a b) c)'))
        ( ( a b ) c )
    '''
    string = string.replace('(', ' ( ')
    string = string.replace(')', ' ) ')
    string = ' '.join(string.split())
    return string
