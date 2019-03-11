import sys
from tab_process import find_max, from_json
from print_table import output, output_h
# Ваши импорты


def get_data(filename):
    if filename:
        codes = ['utf8', 'utf16', 'cp1251']
        for enc in codes:
            try:
                try:
                    with open(filename, encoding=enc) as f_obj:
                        data = from_json(f_obj)
                    if data == 'Формат не валиден':
                        with open(filename, encoding=enc) as f_obj:
                            data = [line.strip().split('\t') for line in f_obj]
                            break
                    else:
                        break
                except FileNotFoundError:
                    return "Файл не валиден"
            except UnicodeError:
                pass
        width, col_numb = find_max(data)
        if not width:
            return "Формат не валиден"
        header = 1
        print('-' * (sum(width) + 5 * col_numb + 1), end='\n')
        for row in data:
            if header:
                print(output_h(row, width), end='')
            else:
                print(output(row, width), end='')
            header = 0
        print('-' * (sum(width) + 5 * col_numb + 1))
    else:
        return "Файл не валиден"

if __name__ == '__main__':
    filename =  sys.argv[1] # "D:/atom/AppliedPythonAtom/homeworks/homework_02/files/posts-utf8.tsv"
    # filename = "D:/atom/AppliedPythonAtom/output.txt"
    answer = get_data(filename)
    if answer:
        print(answer)