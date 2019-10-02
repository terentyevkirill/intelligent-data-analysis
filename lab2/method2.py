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
    x1 = np.arange(center1[0], center2[0], 0.001)
    y1 = line_equation(a1, b1, x1)
    return [x1, y1]


def get_mid_section(dx, dy, coord_x, coord_y):
    x_o = coord_x + abs(dx) / 2
    y_o = coord_y + abs(dy) / 2
    return [x_o, y_o]


def get_separating_line(dx, dy, mid_line_point):
    a = -dx / dy
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
    plt.plot(center1[0], center1[1], '+r')
    plt.plot(center2[0], center2[1], '+b')
    plt.plot(center3[0], center3[1], '+y')
    plt.plot(center4[0], center4[1], '+g')

    # Вычисляем центроиды для всех точек, не принадлежащих данному классу
    class234 = np.append(class2, np.array([class3, class4]))
    class234 = class234.reshape((15, 2))
    center234 = get_centroid(class234)

    class134 = np.append(class1, np.array([class3, class4]))
    class134 = class134.reshape((15, 2))
    center134 = get_centroid(class134)

    class124 = np.append(class1, np.array([class2, class4]))
    class124 = class124.reshape((15, 2))
    center124 = get_centroid(class124)

    class123 = np.append(class1, np.array([class2, class3]))
    class123 = class123.reshape((15, 2))
    center123 = get_centroid(class123)

    # Рисуем центроиды для "остальных точек" к каждому классу
    plt.plot(center234[0], center234[1], '+r')
    plt.plot(center134[0], center134[1], '+b')
    plt.plot(center124[0], center124[1], '+y')
    plt.plot(center123[0], center123[1], '+g')

    # Вычисляем координаты
    dx_1_234 = center234[0] - center1[0]
    dy_1_234 = center234[1] - center1[1]
    dx_2_134 = center134[0] - center2[0]
    dy_2_134 = center134[1] - center2[1]
    dx_3_124 = center124[0] - center3[0]
    dy_3_124 = center124[1] - center3[1]
    dx_4_123 = center123[0] - center4[0]
    dy_4_123 = center123[1] - center4[1]

    # Вычисляем середины отрезков, соединяющих центроиды с центроидами остальных точек
    mid_line_p_1_234 = get_mid_section(dx_1_234, dy_1_234, center1[0], center234[1])
    mid_line_p_2_134 = get_mid_section(dx_2_134, dy_2_134, center134[0], center134[1])
    mid_line_p_3_124 = get_mid_section(dx_3_124, dy_3_124, center124[0], center3[1])
    mid_line_p_4_123 = get_mid_section(dx_4_123, dy_4_123, center4[0], center4[1])

    # Вычисляем линии, проходящие между классами
    line_x_1_234, line_y_1_234 = get_class_connect_line_equation(dx_1_234, dy_1_234, center1, center234)
    line_x_2_134, line_y_2_134 = get_class_connect_line_equation(dx_2_134, dy_2_134, center2, center134)
    line_x_3_124, line_y_3_124 = get_class_connect_line_equation(dx_3_124, dy_3_124, center3, center124)
    line_x_4_123, line_y_4_123 = get_class_connect_line_equation(dx_4_123, dy_4_123, center4, center123)

    # Рисуем линии, проходящие между классами
    # plt.plot(line_x_1_234, line_y_1_234, 'gray')
    # plt.plot(line_x_2_134, line_y_2_134, 'gray')
    # plt.plot(line_x_3_124, line_y_3_124, 'gray')
    # plt.plot(line_x_4_123, line_y_4_123,'gray')

    # Вычисляем линии, разделяющие классы
    sep_line_1_234 = get_separating_line(dx_1_234, dy_1_234, mid_line_p_1_234)
    sep_line_2_134 = get_separating_line(dx_2_134, dy_2_134, mid_line_p_2_134)
    sep_line_3_124 = get_separating_line(dx_3_124, dy_3_124, mid_line_p_3_124)
    sep_line_4_123 = get_separating_line(dx_4_123, dy_4_123, mid_line_p_4_123)

    # Рисуем линии, разделяющие классы
    plt.plot(sep_line_1_234['coord_x'], sep_line_1_234['coord_y'], '-k')
    plt.plot(sep_line_2_134['coord_x'], sep_line_2_134['coord_y'], '-k')
    plt.plot(sep_line_3_124['coord_x'], sep_line_3_124['coord_y'], '-k')
    plt.plot(sep_line_4_123['coord_x'], sep_line_4_123['coord_y'], '-k')

    # def isClass1(point):  # Принадлежит ли точка классу 1
    #     # находим коэффициенты разделяющих линий для класса 1
    #     sep_line_k_12 = [sep_line_1_234['a'], sep_line_1_234['b']]
    #     sep_line_k_13 = [sep_line_2_134['a'], sep_line_2_134['b']]
    #     sep_line_k_14 = [sep_line_3_124['a'], sep_line_3_124['b']]
    #     # находим решающие функции для класса 1
    #     d12 = point[1] - sep_line_k_12[0] * point[0] - sep_line_k_12[1]
    #     d13 = point[1] - sep_line_k_13[0] * point[0] - sep_line_k_13[1]
    #     d14 = point[1] - sep_line_k_14[0] * point[0] - sep_line_k_14[1]
    #     #  проверяем условие
    #     return d12 > 0 and d13 > 0 and d14 > 0
    #
    # def isClass2(point):  # Принадлежит ли точка классу 2
    #     sep_line_k_12 = [sep_line_1_234['a'], sep_line_1_234['b']]
    #     sep_line_k_23 = [sep_line_4_123['a'], sep_line_4_123['b']]
    #     sep_line_k_24 = [sep_line_42['a'], sep_line_42['b']]
    #     d12 = point[1] - sep_line_k_12[0] * point[0] - sep_line_k_12[1]
    #     d23 = point[1] - sep_line_k_23[0] * point[0] - sep_line_k_23[1]
    #     d24 = point[1] - sep_line_k_24[0] * point[0] - sep_line_k_24[1]
    #     return d12 <= 0 and d23 > 0 and d24 > 0
    #
    # def isClass3(point):
    #     sep_line_k_13 = [sep_line_2_134['a'], sep_line_2_134['b']]
    #     sep_line_k_23 = [sep_line_4_123['a'], sep_line_4_123['b']]
    #     sep_line_k_34 = [sep_line_43['a'], sep_line_43['b']]
    #     d13 = point[1] - sep_line_k_13[0] * point[0] - sep_line_k_13[1]
    #     d23 = point[1] - sep_line_k_23[0] * point[0] - sep_line_k_23[1]
    #     d34 = point[1] - sep_line_k_34[0] * point[0] - sep_line_k_34[1]
    #     return d13 <= 0 and d23 <= 0 and d34 <= 0
    #
    # def isClass4(point):
    #     sep_line_k_14 = [sep_line_3_124['a'], sep_line_3_124['b']]
    #     sep_line_k_24 = [sep_line_42['a'], sep_line_42['b']]
    #     sep_line_k_34 = [sep_line_43['a'], sep_line_43['b']]
    #     d14 = point[1] - sep_line_k_14[0] * point[0] - sep_line_k_14[1]
    #     d24 = point[1] - sep_line_k_24[0] * point[0] - sep_line_k_24[1]
    #     d34 = point[1] - sep_line_k_34[0] * point[0] - sep_line_k_34[1]
    #     return d14 <= 0 and d24 <= 0 and d34 > 0

    # def classify(point):
    #     if isClass1(point) and not (isClass2(point) or isClass3(point) or isClass4(point)):
    #         return 'class1'
    #     elif isClass2(point) and not (isClass1(point) or isClass3(point) or isClass4(point)):
    #         return 'class2'
    #     elif isClass3(point) and not (isClass1(point) or isClass2(point) or isClass4(point)):
    #         return 'class3'
    #     elif isClass4(point) and not (isClass1(point) or isClass2(point) or isClass3(point)):
    #         return 'class4'
    #     else:
    #         return 'unclassified'
    #
    # def classifyClasses():
    #     classes = [class1, class2, class3, class4]
    #     matr = np.array([[]])
    #     for c in classes:
    #         i = 0
    #         count1 = 0
    #         count2 = 0
    #         count3 = 0
    #         count4 = 0
    #         for p in c:
    #             if classify(p) == 'class1':
    #                 count1 += 1
    #                 plt.plot(p[0], p[1], 'sr')
    #             elif classify(p) == 'class2':
    #                 count2 += 1
    #                 plt.plot(p[0], p[1], 'Db')
    #             elif classify(p) == 'class3':
    #                 count3 += 1
    #                 plt.plot(p[0], p[1], 'oy')
    #             elif classify(p) == 'class4':
    #                 count4 += 1
    #                 plt.plot(p[0], p[1], '^g')
    #             else:
    #                 plt.plot(p[0], p[1], 'ok')
    #
    #         matr = np.append(matr, np.array([count1, count2, count3, count4]))
    #         i += 1
    #
    #     matr = np.reshape(matr, (4, 4))
    #     print(matr.astype(int))

    # classifyClasses()

    # def classifyMatrix():
    #     matrix = [[round(x, 2), round(y, 2)] for x in np.arange(0, 0.61, 0.01) for y in np.arange(0, 1.02, 0.02)]
    #     for p in matrix:
    #         if classify(p) == 'class1':
    #             plt.plot(p[0], p[1], 'or')
    #         elif classify(p) == 'class2':
    #             plt.plot(p[0], p[1], 'ob')
    #         elif classify(p) == 'class3':
    #             plt.plot(p[0], p[1], 'oy')
    #         elif classify(p) == 'class4':
    #             plt.plot(p[0], p[1], 'og')
    #         else:
    #             plt.plot(p[0], p[1], 'ok')

    # classifyMatrix()

    # Рисуем точки на серединах отрезков между классами - поверх линий
    # plt.plot(mid_line_p_1_234[0], mid_line_p_1_234[1], '*k')
    # plt.plot(mid_line_p_2_134[0], mid_line_p_2_134[1], '*k')
    # plt.plot(mid_line_p_3_124[0], mid_line_p_3_124[1], '*k')
    # plt.plot(mid_line_p_4_123[0], mid_line_p_4_123[1], '*k')

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid(True)
    plt.show()
    print()


if __name__ == "__main__":
    main()
