from math import ceil


def output_h(row, width):
    output_row = ''
    for i in range(len(row)):
        print("|  {:^{w}}  ".format(row[i], w=width[i]), end='')
    print("|")


def output(row, width):
    for i in range(len(row)):
        output_row = "|  {:" + (">" if i == len(row) - 1 else "<") + "{w}}  "
        print(output_row.format(row[i], w=width[i]), end='')
    print("|")
