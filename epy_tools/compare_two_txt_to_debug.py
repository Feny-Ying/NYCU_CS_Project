import argparse


def compare_files(file_a, file_b):
    with open(file_a, 'r', encoding='utf-8') as fa, open(file_b, 'r', encoding='utf-8') as fb:
        lines_a = fa.readlines()
        lines_b = fb.readlines()

    max_lines = max(len(lines_a), len(lines_b))

    with open('diff_results.txt', 'w', encoding='utf-8') as fa_diff:
        for i in range(max_lines):
            line_a = lines_a[i] if i < len(lines_a) else ''
            line_b = lines_b[i] if i < len(lines_b) else ''
            if line_a != line_b:
                fa_diff.write(f"--------------- Line {i + 1} ---------------\n")
                if line_a:
                    fa_diff.write(line_a)
                if line_b:
                    fa_diff.write(line_b)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', type=str, required=True)
    parser.add_argument('-f2', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    ARGS = get_args()
    compare_files(ARGS.f1, ARGS.f2)
