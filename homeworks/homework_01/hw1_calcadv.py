#!/usr/bin/env python
# coding: utf-8
import operator
import re


def is_valid(expr):
    only_ops = re.compile(r'^(?:\s*[\*\+\-\/\(\)]\s*)+$')

    startswith_mult_div = re.compile(r'\s*[\/\*].*')

    unknown_symbols = re.compile(r'[^\s\d\+\-\*\/\(\)\.]')

    absent_operator = re.compile(r'\d\s+\d')

    empty_brackets = re.compile(r'\(\s*\)')

    near_ops = re.compile(r'([\+\-\*\/]\s*[\/\*])|([\/\*]\s*[\*\/])')

    if (not expr or
        '\n' in expr or
        re.fullmatch(startswith_mult_div, expr) or
        re.fullmatch(only_ops, expr) or
        expr.count('(') != expr.count(')') or
        re.search(unknown_symbols, expr) or
        re.search(absent_operator, expr) or
        re.search(empty_brackets, expr) or
            re.search(near_ops, expr)):

        return False
    else:
        return True


def correct_ops(expr):
    '''
    ++ = +
    -- = +
    +- -+ = -
    '''
    near_plus_minus = re.compile(r'([\+\-]\s*[\+\-])')
    while re.search(near_plus_minus, expr):

        while re.search(r'\+\s*\+', expr):
            expr = re.sub(r'\+\s*\+', '+', expr)

        while re.search(r'\-\s*\-', expr):
            expr = re.sub(r'\-\s*\-', '+', expr)

        while re.search(r'(\+\s*\-)|(\-\s*\+)', expr):
            expr = re.sub(r'(\+\s*\-)|(\-\s*\+)', '-', expr)

    if re.search(r'^\s*\+\s*', expr):
        expr = re.sub(r'^\s*\+\s*', '', expr)

    return expr


def infix_to_postfix(string):
    string = string.strip() + ' '
    ops = {'+': 0, '-': 0, '*': 1, '/': 1, '(': -1, ')': -1}
    output = ''
    stack = []
    i = 0
    after_mult_div = False
    if string[0] == '-':
        output += '-'
        string = string[1:]
    while i < len(string) - 1:
        if string[i] in '*/':
            j = i + 1
            while not (string[j].isdigit() or string[j] == '('):
                if string[j] in [' ', '\t']:
                    j += 1
                elif string[j] in '-+':
                    output += ' ' + string[j]
                    string = string[:j] + string[j + 1:]
                    after_mult_div = True

        if string[i] in [' ', '\t']:
            i += 1
            continue

        elif string[i].isdigit() or string[i] == '.':
            output += string[i]
            i += 1

        elif string[i] == '(':
            stack.append(string[i])
            i += 1

        elif string[i] == ')':
            try:
                while stack[-1] != '(':
                    output += ' ' + stack.pop()
            except IndexError:
                return None
            stack.pop()
            i += 1

        elif string[i] in ops:
            while stack and ops[stack[-1]] >= ops[string[i]]:
                output += ' ' + stack.pop()

            stack.append(string[i])
            output += '' if after_mult_div else ' '
            after_mult_div = False
            i += 1

        else:
            return None

    while len(stack) > 0:
        if stack[-1] not in ops:
            return None
        else:
            output += ' ' + stack.pop()

    return output.strip()


def calc(tokens):
    if tokens is None:
        return None
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv}
    stack = []
    for token in tokens.split():
        if token in ops:
            op2, op1 = stack.pop(), stack.pop()
            stack.append(ops[token](op1, op2))
        elif token:
            stack.append(float(token))
    return stack.pop() if len(stack) == 1 else None


def advanced_calculator(input_string):
    '''
    Калькулятор на основе обратной польской записи.
    Разрешенные операции: открытая скобка, закрытая скобка,
     плюс, минус, умножить, делить
    :param input_string: строка, содержащая выражение
    :return: результат выполнение операции, если строка валидная - иначе None
    '''

    if is_valid(input_string):
        return calc(infix_to_postfix(correct_ops(input_string)))
    else:
        return None
