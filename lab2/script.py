import numpy as np
import matplotlib.pyplot as plt

class_red = np.array(
    [[0.05, 0.91],
     [0.14, 0.96],
     [0.16, 0.9],
     [0.07, 0.7],
     [0.2, 0.63]])
class_blue = np.array(
    [[0.49, 0.89],
     [0.34, 0.81],
     [0.36, 0.67],
     [0.47, 0.49],
     [0.52, 0.53]])
class_yellow = np.array(
    [[0.31, 0.43],
     [0.45, 0.27],
     [0.33, 0.16],
     [0.56, 0.29],
     [0.54, 0.13]])
class_green = np.array(
    [[0.05, 0.15],
     [0.09, 0.39],
     [0.13, 0.51],
     [0.25, 0.34],
     [0.15, 0.36]])

def line_equation(a, b, x):
    return a * x + b

def get_centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return [sum(x) / len(points), sum(y) / len(points)]

def get_class_connect_line_equation(dx, dy, center1, center2):
    a1 = dy / dx
    b1 = center1[1] - a1 * center1[0]
    x1 = np.arange(center1[0], center2[0], 0.0001)
    y1 = line_equation(a1, b1, x1)
    return [x1, y1]

def get_mid_section(dx, dy, coord_x, coord_y):
    x_o = coord_x + abs(dx) / 2
    y_o = coord_y + abs(dy) / 2
    return [x_o, y_o]

def get_separating_line(dx, dy, mid_line_point):
    a = -dx / dy
    b = (mid_line_point[0] * dx + mid_line_point[1] * dy) / dy
    coord_x_array = np.arange(0.05, 0.5, 0.01)
    return [coord_x_array, line_equation(a, b, coord_x_array)]



def main():
    # Рисуем классы
    p_red, = plt.plot(class_red[:, 0], class_red[:, 1], 'sr')
    p_blue, = plt.plot(class_blue[:, 0], class_blue[:, 1], 'Db')
    p_yellow, = plt.plot(class_yellow[:, 0], class_yellow[:, 1], 'oy')
    p_green, = plt.plot(class_green[:, 0], class_green[:, 1], '^g')
    plt.legend([p_red, p_blue, p_yellow, p_green], ["Class 1", "Class 2", "Class 3", "Class 4"])

    # Находим центроиды
    center_red = get_centroid(class_red)
    center_blue = get_centroid(class_blue)
    center_yellow = get_centroid(class_yellow)
    center_green = get_centroid(class_green)

    # Рисуем центры классов
    plt.plot(center_red[0], center_red[1], '+r')
    plt.plot(center_blue[0], center_blue[1], '+b')
    plt.plot(center_yellow[0], center_yellow[1], '+y')
    plt.plot(center_green[0], center_green[1], '+g')

    # Вычисляем координаты
    rb_dx = center_blue[0] - center_red[0]  # red-blue dx
    rb_dy = center_blue[1] - center_red[1]  # red-blue dy
    ry_dx = center_yellow[0] - center_red[0]  # red-yellow dx
    ry_dy = center_yellow[1] - center_red[1]  # red-yellow dy
    rg_dx = center_green[0] - center_red[0]  # red-green dx
    rg_dy = center_green[1] - center_red[1] # red-green dy
    by_dx = center_yellow[0] - center_blue[0] # blue-yellow dx
    by_dy = center_yellow[1] - center_blue[1]  # blue-yellow dy
    bg_dx = center_green[0] - center_blue[0] # blue-green dx
    bg_dy = center_green[1] - center_blue[1]  # blue-green dy
    gb_dx = center_blue[0] - center_green[0]
    gb_dy = center_blue[1] - center_green[1]
    yg_dx = center_green[0] - center_yellow[0]  # yellow-green dx
    yg_dy = center_green[1] - center_yellow[1] # yellow-green dy
    gy_dx = center_yellow[0] - center_green[0]
    gy_dy = center_yellow[1] - center_green[1]

    # Вычисляем середины отрезков, соединяющих классы
    rb_mid_line_p = get_mid_section(rb_dx, rb_dy, center_red[0], center_blue[1])
    ry_mid_line_p = get_mid_section(ry_dx, ry_dy, center_red[0], center_yellow[1])
    rg_mid_line_p = get_mid_section(rg_dx, rg_dy, center_red[0], center_green[1])
    by_mid_line_p = get_mid_section(by_dx, by_dy, center_blue[0], center_yellow[1])
    # не работает
    bg_mid_line_p = get_mid_section(bg_dx, bg_dy, center_blue[0], center_green[1])
    gb_mid_line_p = get_mid_section(gb_dx, gb_dy, center_green[0], center_green[1])
    yg_mid_line_p = get_mid_section(yg_dx, yg_dy, center_yellow[0], center_green[1])
    gy_mid_line_p = get_mid_section(gy_dx, gy_dy, center_green[0], center_yellow[1])

    # Вычисляем линию, проходящую между классами
    rb_line_x, rb_line_y = get_class_connect_line_equation(rb_dx, rb_dy, center_red, center_blue)
    ry_line_x, ry_line_y = get_class_connect_line_equation(ry_dx, ry_dy, center_red, center_yellow)
    rg_line_x, rg_line_y = get_class_connect_line_equation(rg_dx, rg_dy, center_red, center_green)
    by_line_x, by_line_y = get_class_connect_line_equation(by_dx, by_dy, center_blue, center_yellow)
    # bg_line_x, bg_line_y = get_class_connect_line_equation(bg_dx, bg_dy, center_blue, center_green)   # соединяет неправильно
    gb_line_x, gb_line_y = get_class_connect_line_equation(gb_dx, gb_dy, center_green, center_blue)     # соединяет классы правильно

    # yg_line_x, yg_line_y = get_class_connect_line_equation(yg_dx, yg_dy, center_yellow, center_green)
    gy_line_x, gy_line_y = get_class_connect_line_equation(gy_dx, gy_dy, center_green, center_yellow)

    plt.plot(rb_line_x, rb_line_y)
    plt.plot(ry_line_x, ry_line_y)
    plt.plot(rg_line_x, rg_line_y)
    plt.plot(by_line_x, by_line_y)
    plt.plot(gb_line_x, gb_line_y)
    # plt.plot(bg_line_x, bg_line_y)
    # plt.plot(yg_line_x, yg_line_y)
    plt.plot(gy_line_x, gy_line_y)

    # Вычисляем линию, разделяющую классы
    rb_sep_line = get_separating_line(rb_dx, rb_dy, rb_mid_line_p)
    ry_sep_line = get_separating_line(ry_dx, ry_dy, ry_mid_line_p)
    rg_sep_line = get_separating_line(rg_dx, rg_dy, rg_mid_line_p)
    by_sep_line = get_separating_line(by_dx, by_dy, by_mid_line_p)
    # bg_sep_line = get_separating_line(bg_dx, bg_dy, bg_mid_line_p)
    gb_sep_line = get_separating_line(gb_dx, gb_dy, gb_mid_line_p)
    # yg_sep_line = get_separating_line(yg_dx, yg_dy, yg_mid_line_p)
    gy_sep_line = get_separating_line(gy_dx, gy_dy, gy_mid_line_p)

    plt.plot(rb_sep_line[0], rb_sep_line[1])
    plt.plot(ry_sep_line[0], ry_sep_line[1])
    plt.plot(rg_sep_line[0], rg_sep_line[1])
    plt.plot(by_sep_line[0], by_sep_line[1])
    plt.plot(gb_sep_line[0], gb_sep_line[1])
    # plt.plot(bg_sep_line[0], bg_sep_line[1])
    # plt.plot(yg_sep_line[0], yg_sep_line[1])
    plt.plot(gy_sep_line[0], gy_sep_line[1])

    # Рисуем точки на серединах отрезков между классами - поверх линий
    plt.plot(rb_mid_line_p[0], rb_mid_line_p[1], '*k')
    plt.plot(ry_mid_line_p[0], ry_mid_line_p[1], '*k')
    plt.plot(rg_mid_line_p[0], rg_mid_line_p[1], '*k')
    plt.plot(by_mid_line_p[0], by_mid_line_p[1], '*k')
    # plt.plot(bg_mid_line_p[0], bg_mid_line_p[1], '*k')
    plt.plot(gb_mid_line_p[0], gb_mid_line_p[1], '*k')
    # plt.plot(yg_mid_line_p[0], yg_mid_line_p[1], '*k')
    plt.plot(gy_mid_line_p[0], gy_mid_line_p[1], '*k')

    plt.grid(True)
    plt.show()
    print()


if __name__ == "__main__":
    main()