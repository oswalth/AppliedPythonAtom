#!/usr/bin/env python
# coding: utf-8


def is_bracket_correct(input_string):
    '''
    Метод проверяющий является ли поданная скобочная
     последовательность правильной (скобки открываются и закрываются)
     не пересекаются
    :param input_string: строка, содержащая 6 типов скобок (,),[,],{,}
    :return: True or False
    '''
    brs_dict = {'{': '}', '[': ']', '(': ')'}
    close = (')', ']', '}')
    stack = []
    counter = 0
    for el in input_string:
        if el in close and len(stack) == 0:
            return False
        elif el in close:
            if el == brs_dict.get(stack.pop()):
                counter -= 1
        else:
            stack.append(el)
            counter += 1
    return not bool(counter)


b = is_bracket_correct(']{[})(')
print(b)
