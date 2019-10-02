import numpy as np
import matplotlib.pyplot as plt

class1 = np.array([[0.05, 0.91],
                   [0.14, 0.96],
                   [0.16, 0.9],
                   [0.07, 0.7],
                   [0.2, 0.63]])
class2 = np.array([[0.49, 0.89],
                   [0.34, 0.81],
                   [0.36, 0.67],
                   [0.47, 0.49],
                   [0.52, 0.53]])
class3 = np.array([[0.31, 0.43],
                   [0.45, 0.27],
                   [0.33, 0.16],
                   [0.56, 0.29],
                   [0.54, 0.13]])
class4 = np.array([[0.05, 0.15],
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
    a = -dx / dys
    b = (mid_line_point[0] * dx + mid_line_point[1] * dy) / dy
    coord_x_array = np.arange(0, 0.61, 0.01)
    return {
        'coord_x': coord_x_array,
        'coord_y': line_equation(a, b, coord_x_array),
        'a': a,
        'b': b
    }


def main():
    # Рисуем классы
    # p1, = plt.plot(class1[:, 0], class1[:, 1], 'sr')
    # p2, = plt.plot(class2[:, 0], class2[:, 1], 'Db')
    # p3, = plt.plot(class3[:, 0], class3[:, 1], 'oy')
    # p4, = plt.plot(class4[:, 0], class4[:, 1], '^g')
    # plt.legend([p1, p2, p3, p_green], ["Class 1", "Class 2", "Class 3", "Class 4"])

    # Находим центроиды
    center1 = get_centroid(class1)
    center2 = get_centroid(class2)
    center3 = get_centroid(class3)
    center4 = get_centroid(class4)

    # Рисуем центры классов
    # plt.plot(center1[0], center1[1], '+r')
    # plt.plot(center2[0], center2[1], '+b')
    # plt.plot(center3[0], center3[1], '+y')
    # plt.plot(center4[0], center4[1], '+g')

    class234 = np.append(class2, (class3, class4))
    print(class234)

    # Вычисляем координаты
    dx_12 = center2[0] - center1[0]  # red-blue dx
    dy_12 = center2[1] - center1[1]  # red-blue dy
    dx_13 = center3[0] - center1[0]  # red-yellow dx
    dy_13 = center3[1] - center1[1]  # red-yellow dy
    dx_14 = center4[0] - center1[0]  # red-green dx
    dy_14 = center4[1] - center1[1]  # red-green dy
    dx_23 = center3[0] - center2[0]  # blue-yellow dx
    dy_23 = center3[1] - center2[1]  # blue-yellow dy
    dx_42 = center2[0] - center4[0]  # green-blue dx
    dy_42 = center2[1] - center4[1]  # green-blue dy
    dx_43 = center3[0] - center4[0]  # green-yellow dx
    dy_43 = center3[1] - center4[1]  # green-yellow dy

    # Вычисляем середины отрезков, соединяющих классы
    mid_line_p_12 = get_mid_section(dx_12, dy_12, center1[0], center2[1])
    mid_line_p_13 = get_mid_section(dx_13, dy_13, center1[0], center3[1])
    mid_line_p_14 = get_mid_section(dx_14, dy_14, center1[0], center4[1])
    mid_line_p_23 = get_mid_section(dx_23, dy_23, center2[0], center3[1])
    mid_line_p_42 = get_mid_section(dx_42, dy_42, center4[0], center4[1])
    mid_line_p_43 = get_mid_section(dx_43, dy_43, center4[0], center3[1])

    # Вычисляем линии, проходящие между классами
    line_x_12, line_y_12 = get_class_connect_line_equation(dx_12, dy_12, center1, center2)
    line_x_13, line_y_13 = get_class_connect_line_equation(dx_13, dy_13, center1, center3)
    line_x_14, line_y_14 = get_class_connect_line_equation(dx_14, dy_14, center1, center4)
    line_x_23, line_y_23 = get_class_connect_line_equation(dx_23, dy_23, center2, center3)
    line_x_42, line_y_42 = get_class_connect_line_equation(dx_42, dy_42, center4, center2)
    line_x_43, line_y_43 = get_class_connect_line_equation(dx_43, dy_43, center4, center3)

    # Рисуем линии, проходящие между классами
    # plt.plot(line_x_12, line_y_12, 'gray')
    # plt.plot(line_x_13, line_y_13, 'gray')
    # plt.plot(line_x_14, line_y_14, 'gray')
    # plt.plot(line_x_23, line_y_23,'gray')
    # plt.plot(line_x_42, line_y_42,'gray')
    # plt.plot(line_x_43, line_y_43,'gray')

    # Вычисляем линии, разделяющие классы
    sep_line_12 = get_separating_line(dx_12, dy_12, mid_line_p_12)
    sep_line_13 = get_separating_line(dx_13, dy_13, mid_line_p_13)
    sep_line_14 = get_separating_line(dx_14, dy_14, mid_line_p_14)
    sep_line_23 = get_separating_line(dx_23, dy_23, mid_line_p_23)
    sep_line_42 = get_separating_line(dx_42, dy_42, mid_line_p_42)
    sep_line_43 = get_separating_line(dx_43, dy_43, mid_line_p_43)
    sep_lines = [[x['a'], x['b']] for x in [sep_line_12, sep_line_13, sep_line_14, sep_line_23, sep_line_42,
                                            sep_line_43]]  # координаты A, B разделяющих линий

    def isClass1(point):  # Принадлежит ли точка классу 1
        # находим коэффициенты разделяющих линий для класса 1
        sep_line_k_12 = [sep_line_12['a'], sep_line_12['b']]
        sep_line_k_13 = [sep_line_13['a'], sep_line_13['b']]
        sep_line_k_14 = [sep_line_14['a'], sep_line_14['b']]
        # находим решающие функции для класса 1
        d12 = point[1] - sep_line_k_12[0] * point[0] - sep_line_k_12[1]
        d13 = point[1] - sep_line_k_13[0] * point[0] - sep_line_k_13[1]
        d14 = point[1] - sep_line_k_14[0] * point[0] - sep_line_k_14[1]
        #  проверяем условие
        return d12 > 0 and d13 > 0 and d14 > 0

    def isClass2(point):  # Принадлежит ли точка классу 2
        sep_line_k_12 = [sep_line_12['a'], sep_line_12['b']]
        sep_line_k_23 = [sep_line_23['a'], sep_line_23['b']]
        sep_line_k_24 = [sep_line_42['a'], sep_line_42['b']]
        d12 = point[1] - sep_line_k_12[0] * point[0] - sep_line_k_12[1]
        d23 = point[1] - sep_line_k_23[0] * point[0] - sep_line_k_23[1]
        d24 = point[1] - sep_line_k_24[0] * point[0] - sep_line_k_24[1]
        return d12 <= 0 and d23 > 0 and d24 > 0

    def isClass3(point):
        sep_line_k_13 = [sep_line_13['a'], sep_line_13['b']]
        sep_line_k_23 = [sep_line_23['a'], sep_line_23['b']]
        sep_line_k_34 = [sep_line_43['a'], sep_line_43['b']]
        d13 = point[1] - sep_line_k_13[0] * point[0] - sep_line_k_13[1]
        d23 = point[1] - sep_line_k_23[0] * point[0] - sep_line_k_23[1]
        d34 = point[1] - sep_line_k_34[0] * point[0] - sep_line_k_34[1]
        return d13 <= 0 and d23 <= 0 and d34 <= 0

    def isClass4(point):
        sep_line_k_14 = [sep_line_14['a'], sep_line_14['b']]
        sep_line_k_24 = [sep_line_42['a'], sep_line_42['b']]
        sep_line_k_34 = [sep_line_43['a'], sep_line_43['b']]
        d14 = point[1] - sep_line_k_14[0] * point[0] - sep_line_k_14[1]
        d24 = point[1] - sep_line_k_24[0] * point[0] - sep_line_k_24[1]
        d34 = point[1] - sep_line_k_34[0] * point[0] - sep_line_k_34[1]
        return d14 <= 0 and d24 <= 0 and d34 > 0


    def classify(point):
        if isClass1(point) and not (isClass2(point) or isClass3(point) or isClass4(point)):
            return 'class1'
        elif isClass2(point) and not (isClass1(point) or isClass3(point) or isClass4(point)):
            return 'class2'
        elif isClass3(point) and not (isClass1(point) or isClass2(point) or isClass4(point)):
            return 'class3'
        elif isClass4(point) and not (isClass1(point) or isClass2(point) or isClass3(point)):
            return 'class4'
        else:
            return 'unclassified'

    def classifyClasses():
        classes = [class1, class2, class3, class4]
        matr = np.array([[]])
        for c in classes:
            i = 0
            count1 = 0
            count2 = 0
            count3 = 0
            count4 = 0
            for p in c:
                if classify(p) == 'class1':
                    count1 += 1
                    plt.plot(p[0], p[1], 'sr')
                elif classify(p) == 'class2':
                    count2 += 1
                    plt.plot(p[0], p[1], 'Db')
                elif classify(p) == 'class3':
                    count3 += 1
                    plt.plot(p[0], p[1], 'oy')
                elif classify(p) == 'class4':
                    count4 += 1
                    plt.plot(p[0], p[1], '^g')
                else:
                    plt.plot(p[0], p[1], 'ok')

            matr = np.append(matr, np.array([count1, count2, count3, count4]))
            i += 1

        matr = np.reshape(matr, (4, 4))
        print(matr.astype(int))

    # classifyClasses()

    def classifyMatrix():
        matrix = [[round(x, 2), round(y, 2)] for x in np.arange(0, 0.61, 0.01) for y in np.arange(0, 1.02, 0.02)]
        for p in matrix:
            if classify(p) == 'class1':
                plt.plot(p[0], p[1], 'or')
            elif classify(p) == 'class2':
                plt.plot(p[0], p[1], 'ob')
            elif classify(p) == 'class3':
                plt.plot(p[0], p[1], 'oy')
            elif classify(p) == 'class4':
                plt.plot(p[0], p[1], 'og')
            else:
                plt.plot(p[0], p[1], 'ok')

    # classifyMatrix()

    # Рисуем линии, разделяющие классы
    # plt.plot(sep_line_12['coord_x'], sep_line_12['coord_y'], '-k')
    # plt.plot(sep_line_13['coord_x'], sep_line_13['coord_y'], '-k')
    # plt.plot(sep_line_14['coord_x'], sep_line_14['coord_y'], '-k')
    # plt.plot(sep_line_23['coord_x'], sep_line_23['coord_y'], '-k')
    # plt.plot(sep_line_42['coord_x'], sep_line_42['coord_y'], '-k')
    # plt.plot(sep_line_43['coord_x'], sep_line_43['coord_y'], '-k')

    # Рисуем точки на серединах отрезков между классами - поверх линий
    # plt.plot(rb_mid_line_p[0], rb_mid_line_p[1], '*k')
    # plt.plot(ry_mid_line_p[0], ry_mid_line_p[1], '*k')
    # plt.plot(rg_mid_line_p[0], rg_mid_line_p[1], '*k')
    # plt.plot(by_mid_line_p[0], by_mid_line_p[1], '*k')
    # plt.plot(gb_mid_line_p[0], gb_mid_line_p[1], '*k')
    # plt.plot(gy_mid_line_p[0], gy_mid_line_p[1], '*k')
    plt.xlim((0, 0.6))
    plt.ylim((0, 1))
    plt.grid(True)
    plt.show()
    print()


if __name__ == "__main__":
    main()
