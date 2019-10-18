'''
Вариант 3
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    draw_line(x_arr, m, c, '-r')
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 20])
    plt.text(0, 21, 'Start m={0}; c={1}'.format(m, c))
    plt.grid(True)
    plt.ion()
    plt.show()
    plt.pause(2)


def update_plot(x_arr, y_arr, m_init, c_init, m, c, i):
    plt.pause(0.001)
    plt.clf()
    axes = plt.gca()
    plt.grid(True)
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 20])
    plt.text(0, 21, 'Iteration {0}'.format(i))
    plt.text(2.5, 21, 'm={0}; c={1}'.format(m, c))
    draw_points(x_arr, y_arr, '.k')
    draw_line(x_arr, m_init, c_init, '-r')
    draw_line(x_arr, m, c)
    plt.show()


def draw_errors(e_arr, e_min):
    plt.pause(3)
    plt.clf()
    plt.grid(True)

    p1 = plt.axhline(e_min, 0, len(e_arr), color='red')
    p2, = plt.plot(e_arr)
    plt.legend([p1, p2], ['Least squares method error', 'Neural network method error'])
    plt.xlabel('Iterations ({0})'.format(len(e_arr)))
    plt.ylabel('Error')


def neural(x_arr, y_arr, m_init, c_init, n_coef, e_min, n_max):
    m = np.random.uniform(0, 5)
    c = np.random.uniform(0, 5)
    e_arr = np.array([])
    m_arr = np.array([])
    c_arr = np.array([])
    e2d_arr = np.array([])
    n = 1  # outer iteration counter
    while True:
        error = 0
        for i in range(len(x_arr)):
            dy = get_error(y_arr[i], m * x_arr[i] + c)  # сигма, ошибка
            dm = dy * x_arr[i] * n_coef
            dc = dy * n_coef
            m += dm
            c += dc
            error += get_error(y_arr[i], m * x_arr[i] + c) ** 2

        m_arr = np.append(m_arr, m)
        c_arr = np.append(c_arr, c)
        e_arr = np.append(e_arr, error)
        update_plot(x_arr, y_arr, m_init, c_init, m, c, n)
        n += 1
        if e_arr[-1] <= e_min or n > n_max:
            break

    for m in m_arr:
        for c in c_arr:
            e2d_arr = np.append(e2d_arr, sum(y_arr - (m * x_arr + c)))
    return [m_arr, c_arr, e_arr, e2d_arr]


def main():
    x_arr = np.array([i + 1 for i in range(9)])
    y_arr = np.array([5.8, 7.8, 8, 9.4, 10.2, 11, 12, 13, 14.4])
    n_coef = 0.01
    n_max = 300
    m_init, c_init, e_min = least_squares(x_arr, y_arr)
    print("Min Error={0}".format(e_min))
    print("Least square: m={0}; c={1}; e={2}".format(m_init, c_init, e_min))
    init_plot(x_arr, y_arr, m_init, c_init)
    m_arr, c_arr, e_arr, e2d_arr = neural(x_arr, y_arr, m_init, c_init, n_coef, e_min, n_max)
    print("Neural network: m={0}; c={1}; e={2}".format(m_arr[-1], c_arr[-1], e_arr[-1]))
    draw_errors(e_arr, e_min)

    plt.pause(5)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    Axes3D.plot_surface(ax, m_arr, c_arr, e2d_arr)
    plt.show()
    plt.pause(100)


if __name__ == '__main__':
    main()
