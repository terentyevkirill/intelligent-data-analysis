'''
Вариант 3
'''
import numpy as np
import matplotlib.pyplot as plt
import random


def draw_points(x_arr, y_arr, *args):
    plt.plot(x_arr, y_arr, *args)


def least_squares(x_arr, y_arr):
    # y = mx + c
    n = len(x_arr)
    # m an c are coefficients
    m = (n * sum(x_arr * y_arr) - sum(x_arr) * sum(y_arr)) / (n * sum(x_arr ** 2) - sum(x_arr) ** 2)
    c = (sum(y_arr) - m * sum(x_arr)) / n
    e = sum((y_arr - (m * x_arr + c)) ** 2)
    return [m, c, e]


def draw_line(x_arr, m, c, *args):
    y_arr = [m * x + c for x in x_arr]
    draw_points(x_arr, y_arr, *args)


def get_error(t, y):
    return t - y


def init_plot(x_arr, y_arr, m, c):
    draw_points(x_arr, y_arr, '.k')
    draw_line(x_arr, m, c)
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 20])
    plt.text(0, 21, 'Start')
    plt.grid(True)
    plt.ion()
    plt.show()


def update_plot(x_arr, y_arr, m_init, c_init, m, c, i):
    plt.pause(0.001)
    plt.clf()
    axes = plt.gca()
    plt.grid(True)
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 20])
    plt.text(5, 21, 'Iteration ' + str(i))
    draw_points(x_arr, y_arr, '.k')
    draw_line(x_arr, m_init, c_init, '-r')
    draw_line(x_arr, m, c)
    plt.show()


def draw_errors(e_arr, e_min):
    plt.pause(3)
    plt.clf()
    plt.grid(True)
    plt.plot(e_arr)
    p1 = plt.axhline(e_min, 0, len(e_arr), color='red')
    plt.legend([p1], ['Least squares method error'])
    plt.xlabel('Iterations')
    plt.ylabel('Error')


def neural(x_arr, y_arr, m_init, c_init, n_coef, e_min, n_max):
    m = np.random.uniform(-50, 50)
    c = np.random.uniform(-50, 50)
    e_arr = np.array([])
    n = 1  # outer iteration counter
    while True:
        error = 0
        for i in range(len(x_arr)):
            dy = get_error(y_arr[i], m * x_arr[i] + c)  # сигма, ошибка
            dm = dy * x_arr[i] * n_coef
            dc = dy * n_coef
            print("dm={0}; dc={1}".format(dm, dc))
            m += dm
            c += dc
            error += get_error(y_arr[i], m * x_arr[i] + c)

        e_arr = np.append(e_arr, abs(error))
        update_plot(x_arr, y_arr, m_init, c_init, m, c, n)
        n += 1
        if e_arr[-1] <= e_min or n > n_max:
            break

    plt.text(0, 21, 'Finish')
    plt.show()
    return [m, c, e_arr]


def main():
    x_arr = np.array([i + 1 for i in range(9)])
    y_arr = np.array([5.8, 7.8, 8, 9.4, 10.2, 11, 12, 13, 14.4])
    n_coef = 0.01
    n_max = 300
    m_init, c_init, e_min = least_squares(x_arr, y_arr)
    print("Min Error={0}".format(e_min))
    print("init m={0}; init c={1}".format(m_init, c_init))
    init_plot(x_arr, y_arr, m_init, c_init)
    m, c, e_arr = neural(x_arr, y_arr, m_init, c_init, n_coef, e_min, n_max)
    print("m={0}; c={1}".format(m, c))
    print("Errors:")
    print(e_arr)
    draw_errors(e_arr, e_min)

    plt.pause(100)


if __name__ == '__main__':
    main()
