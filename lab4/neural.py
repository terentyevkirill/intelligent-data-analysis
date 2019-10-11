'''
Вариант 3
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization


def draw_line(x_arr, y_arr):
    plt.plot(x_arr, y_arr)


def least_squares(x_arr, y_arr):
    # y = mx + c
    n = len(x_arr)
    m = (n * sum(x_arr * y_arr) - sum(x_arr) * sum(y_arr)) / (n * sum(x_arr ** 2) - sum(x_arr) ** 2)
    c = (sum(y_arr) - m * sum(x_arr)) / n
    return {
        'm': m,
        'c': c
    }


def draw_fun_line(x_arr, fun):
    y_arr = [fun['m'] * x + fun['c'] for x in x_arr]
    draw_line(x_arr, y_arr)


def main():
    x_arr = np.array([i + 1 for i in range(9)])
    y_arr = np.array([5.8, 7.8, 8, 9.4, 10.2, 11, 12, 13, 14.4])
    # print(f'x_arr: {x_arr}')
    # print(f'y_arr: {y_arr}')
    draw_line(x_arr, y_arr)
    print(least_squares(x_arr, y_arr))
    draw_fun_line(x_arr, least_squares(x_arr, y_arr))
    plt.show()


if __name__ == '__main__':
    main()
