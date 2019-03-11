from math import ceil


def output_h(row, width):
    output_row = ''
    for i in range(len(row)):
        tmp1 = (width[i] - len(row[i])) / 2
        tmp2 = round(tmp1)
        tmp3 = int(tmp1)
        if i == (len(row) - 1):
            output_row += ('|  ' + ' ' * int((width[i] - len(row[i])) / 2) +
                    row[i] + ' ' * (ceil((width[i] - len(row[i])) / 2) + 2) + '|\n')
        else:
            output_row += ('|  ' + ' ' * int((width[i] - len(row[i])) / 2) +
                row[i] + ' ' * (ceil((width[i] - len(row[i])) / 2) + 2))
    return output_row

def output(row, width):
    output_row = ''
    for i in range(len(row)):
        if i == (len(row) - 1):
            output_row += ('|  ' + ' ' * (width[i] - len(row[i])) + row[i] + "  |\n")
        else:
            output_row += ('|  ' + row[i] + ' ' * (width[i] - len(row[i])) + "  ")
    return output_row