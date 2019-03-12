from math import ceil


def output_h(row, width):
    output_row = ''
    for i in range(len(row)):
        """
        if i == (len(row) - 1):
            output_row += ('|  ' + ' ' * int((width[i] - len(row[i])) / 2) +
                    row[i] + ' ' * (ceil((width[i] - len(row[i])) / 2) + 2) + '|\n')
        else:
            output_row += ('|  ' + ' ' * int((width[i] - len(row[i])) / 2) +
                row[i] + ' ' * (ceil((width[i] - len(row[i])) / 2) + 2))
        """
        print("|  {:^{w}}  ".format(row[i], w=width[i]), end='')
    print("|")

def output(row, width):
    for i in range(len(row)):
        """
        if i == (len(row) - 1):
            output_row += ('|  ' + ' ' * (width[i] - len(row[i])) + row[i] + "  |\n")
        else:
            output_row += ('|  ' + row[i] + ' ' * (width[i] - len(row[i])) + "  ")
        """
        output_row = "|  {:" + (">" if i == len(row) - 1 else "<") + "{w}}  "
        print(output_row.format(row[i], w=width[i]), end='')
    print("|")
