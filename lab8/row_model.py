import numpy as np
import math
from math import exp
import constants as const
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy.linalg as lng


def get_error(s, a=0, b=0, c=0, d=0):
    error = [si - a - b * ti - c * ti ** 2 - d * ti ** 3 for si in s for ti in const.arange]
    # print(f"error={error}")

    return sum(np.power(error, 2))


def get_matrix_a():
    sum_t = sum(const.arange)
    sum_t2 = sum(np.power(const.arange, 2))
    sum_t3 = sum(np.power(const.arange, 3))
    sum_t4 = sum(np.power(const.arange, 4))
    sum_t5 = sum(np.power(const.arange, 5))
    sum_t6 = sum(np.power(const.arange, 6))

    return np.array([
        [const.n, sum_t, sum_t2, sum_t3],
        [sum_t, sum_t2, sum_t3, sum_t4],
        [sum_t2, sum_t3, sum_t4, sum_t5],
        [sum_t3, sum_t4, sum_t5, sum_t6]
    ])


def get_result_matrix(matrix_a, matrix_b, n=4):
    mA = matrix_a[0:n, 0:n]
    mB = matrix_b[0:n]
    return lng.solve(mA, mB)


def get_matrix_b(s):
    sum_s = sum(s)
    sum_s_t = sum(np.multiply(s, const.arange))
    sum_s_t2 = sum(np.multiply(s, np.power(const.arange, 2)))
    sum_s_t3 = sum(np.multiply(s, np.power(const.arange, 3)))
    return np.array([sum_s, sum_s_t, sum_s_t2, sum_s_t3])


def get_row(a=0, b=0, c=0, d=0):
    return [a + b * t + c * t ** 2 + d * t ** 3 for t in const.arange]

def without_noize(s1, s2, s3):
    # WITHOUT NOIZE
    print(f"s1={s1}")
    print(f"s2={s2}")
    print(f"s3={s3}")

    print(f"n={const.n}")

    matrix_a = get_matrix_a()
    # print(f"Matrix a={matrix_a}")
    matrix_b_s1 = get_matrix_b(s1)
    matrix_b_s2 = get_matrix_b(s2)
    matrix_b_s3 = get_matrix_b(s3)
    # print(f"Matrix b_s1={matrix_b_s1}")
    # print(f"Matrix b_s2={matrix_b_s2}")
    # print(f"Matrix b_s3={matrix_b_s3}")

    matrix_x_s11 = get_result_matrix(matrix_a, matrix_b_s1, 2)
    matrix_x_s12 = get_result_matrix(matrix_a, matrix_b_s1, 3)
    matrix_x_s13 = get_result_matrix(matrix_a, matrix_b_s1)
    # print(f"matrix_x_s11={matrix_x_s11}")
    # print(f"matrix_x_s12={matrix_x_s12}")
    # print(f"matrix_x_s13={matrix_x_s13}")

    matrix_x_s21 = get_result_matrix(matrix_a, matrix_b_s2, 2)
    matrix_x_s22 = get_result_matrix(matrix_a, matrix_b_s2, 3)
    matrix_x_s23 = get_result_matrix(matrix_a, matrix_b_s2)

    matrix_x_s31 = get_result_matrix(matrix_a, matrix_b_s3, 2)
    matrix_x_s32 = get_result_matrix(matrix_a, matrix_b_s3, 3)
    matrix_x_s33 = get_result_matrix(matrix_a, matrix_b_s3)

    s11 = get_row(matrix_x_s11[0], matrix_x_s11[1])
    s12 = get_row(matrix_x_s12[0], matrix_x_s12[1], matrix_x_s12[2])
    s13 = get_row(matrix_x_s13[0], matrix_x_s13[1], matrix_x_s13[2], matrix_x_s13[3])

    s21 = get_row(matrix_x_s21[0], matrix_x_s21[1])
    s22 = get_row(matrix_x_s22[0], matrix_x_s22[1], matrix_x_s22[2])
    s23 = get_row(matrix_x_s23[0], matrix_x_s23[1], matrix_x_s23[2], matrix_x_s23[3])

    s31 = get_row(matrix_x_s31[0], matrix_x_s31[1])
    s32 = get_row(matrix_x_s32[0], matrix_x_s32[1], matrix_x_s32[2])
    s33 = get_row(matrix_x_s33[0], matrix_x_s33[1], matrix_x_s33[2], matrix_x_s33[3])

    plt.figure()
    plt.plot(const.arange, s1, color="red")
    plt.plot(const.arange, s2, color="green")
    plt.plot(const.arange, s3, color="blue")
    plt.legend(["s1", "s2", "s3"])
    plt.ylim([0, 100])
    plt.xlabel("t")
    plt.ylabel("s")
    plt.show()

    plt.figure()
    plt.plot(const.arange, s1, color="red")
    plt.plot(const.arange, s11, color="green")
    plt.plot(const.arange, s12, color="blue")
    plt.plot(const.arange, s13, color="yellow")
    plt.xlabel("t")
    plt.ylim([0, 100])
    plt.ylabel("s1")
    plt.legend(["s1", "s11", "s12", "s13"])
    plt.show()

    plt.figure()
    plt.plot(const.arange, s2, color="red")
    plt.plot(const.arange, s21, color="green")
    plt.plot(const.arange, s22, color="blue")
    plt.plot(const.arange, s23, color="yellow")
    plt.xlabel("t")
    plt.ylim([0, 100])
    plt.ylabel("s2")
    plt.legend(["s2", "s21", "s22", "s23"])
    plt.show()

    plt.figure()
    plt.plot(const.arange, s3, color="red")
    plt.plot(const.arange, s31, color="green")
    plt.plot(const.arange, s32, color="blue")
    plt.ylim([0, 100])
    plt.plot(const.arange, s33, color="yellow")
    plt.xlabel("t")
    plt.ylabel("s3")
    plt.legend(["s3", "s31", "s32", "s33"])
    plt.show()

    e11 = get_error(s1, matrix_x_s11[0], matrix_x_s11[1])
    e12 = get_error(s1, matrix_x_s12[0], matrix_x_s12[1], matrix_x_s12[2])
    e13 = get_error(s1, matrix_x_s13[0], matrix_x_s13[1], matrix_x_s13[2], matrix_x_s13[3])
    # WHY MAX?
    min_e1 = max([e11, e12, e13])
    i_min_e1 = np.argmax([e11, e12, e13])

    e21 = get_error(s2, matrix_x_s21[0], matrix_x_s21[1])
    e22 = get_error(s2, matrix_x_s22[0], matrix_x_s22[1], matrix_x_s22[2])
    e23 = get_error(s2, matrix_x_s23[0], matrix_x_s23[1], matrix_x_s23[2], matrix_x_s23[3])
    min_e2 = max([e21, e22, e23])
    i_min_e2 = np.argmax([e21, e22, e23])

    e31 = get_error(s3, matrix_x_s31[0], matrix_x_s31[1])
    e32 = get_error(s3, matrix_x_s32[0], matrix_x_s32[1], matrix_x_s32[2])
    e33 = get_error(s3, matrix_x_s33[0], matrix_x_s33[1], matrix_x_s33[2], matrix_x_s33[3])
    min_e3 = max([e31, e32, e33])
    i_min_e3 = np.argmax([e31, e32, e33])

    print(f"e11={e11}, e12={e12}, e13={e13}, min_e1=e1{i_min_e1 + 1}={min_e1}")
    print(f"e21={e21}, e22={e22}, e23={e23}, min_e2=e2{i_min_e2 + 1}={min_e2}")
    print(f"e31={e31}, e32={e32}, e13={e33}, min_e3=e3{i_min_e3 + 1}={min_e3}")

    res_s1 = [s11, s12, s13][i_min_e1]
    res_s2 = [s21, s22, s23][i_min_e2]
    res_s3 = [s31, s32, s33][i_min_e3]

    plt.figure()
    plt.plot(const.arange, res_s1, color="red")
    plt.plot(const.arange, res_s2, color="green")
    plt.plot(const.arange, res_s3, color="blue")
    plt.legend([f"s1{i_min_e1 + 1}", f"s2{i_min_e2 + 1}", f"s3{i_min_e3 + 1}"])
    plt.ylim([0, 100])
    plt.xlabel("t")
    plt.ylabel("result s")
    plt.show()


def with_noise(s1, s2, s3):
    s1 = get_noised_row(s1)
    s2 = get_noised_row(s2)
    s3 = get_noised_row(s3)

    matrix_a = get_matrix_a()
    matrix_b_s1 = get_matrix_b(s1)
    matrix_b_s2 = get_matrix_b(s2)
    matrix_b_s3 = get_matrix_b(s3)

    matrix_x_s11 = get_result_matrix(matrix_a, matrix_b_s1, 2)
    matrix_x_s22 = get_result_matrix(matrix_a, matrix_b_s2, 3)
    matrix_x_s33 = get_result_matrix(matrix_a, matrix_b_s3)

    s11 = get_row(matrix_x_s11[0], matrix_x_s11[1])
    s22 = get_row(matrix_x_s22[0], matrix_x_s22[1], matrix_x_s22[2])
    s33 = get_row(matrix_x_s33[0], matrix_x_s33[1], matrix_x_s33[2], matrix_x_s33[3])

    plt.figure()
    plt.plot(const.arange, s1, color="red")
    plt.plot(const.arange, s2, color="green")
    plt.plot(const.arange, s3, color="blue")
    plt.legend(["s1", "s2", "s3"])
    plt.ylim([0, 100])
    plt.xlabel("t")
    plt.ylabel("s (noised)")
    plt.show()

    plt.figure()
    plt.plot(const.arange, s1, color="red")
    plt.plot(const.arange, s11, color="green")
    plt.xlabel("t")
    plt.ylim([0, 100])
    plt.ylabel("s1 (noised)")
    plt.legend(["s1", "s11", "s12", "s13"])
    plt.show()

    plt.figure()
    # plt.plot(const.arange, s2, color="red")
    plt.plot(const.arange, s22, color="blue")
    plt.xlabel("t")
    plt.ylim([0, 100])
    plt.ylabel("s2 (noised)")
    plt.legend(["s2", "s21", "s22", "s23"])
    plt.show()

    plt.figure()
    # plt.plot(const.arange, s3, color="red")
    plt.ylim([0, 100])
    plt.plot(const.arange, s33, color="yellow")
    plt.xlabel("t")
    plt.ylabel("s3 (noised)")
    plt.legend(["s3", "s31", "s32", "s33"])
    plt.show()

    e11 = get_error(s1, matrix_x_s11[0], matrix_x_s11[1])
    min_e1 = e11
    i_min_e1 = 0

    e22 = get_error(s2, matrix_x_s22[0], matrix_x_s22[1], matrix_x_s22[2])
    min_e2 = e22
    i_min_e2 = 1

    e33 = get_error(s3, matrix_x_s33[0], matrix_x_s33[1], matrix_x_s33[2], matrix_x_s33[3])
    min_e3 = e33
    i_min_e3 = 2

    print(f"e11={e11}, e22={e22}, e33={e33}")

    plt.figure()
    plt.plot(const.arange, s11, color="red")
    plt.plot(const.arange, s22, color="green")
    plt.plot(const.arange, s33, color="blue")
    plt.legend([f"s1{i_min_e1 + 1}", f"s2{i_min_e2 + 1}", f"s3{i_min_e3 + 1}"])
    plt.ylim([0, 100])
    plt.xlabel("t")
    plt.ylabel("result s (noised)")
    plt.show()


# s4(t)=exp(-bt)+exp(-ct).
def s4(b, c, t):
    return exp(-b * t) + exp(-c * t)

def exponential():
    b = const.b
    c = const.c
    array_b = np.arange(b - 1, b + 1, 0.0625)
    nb = len(array_b)
    array_c = np.arange(c - 2, c + 2, 0.125)
    nc = len(array_c)
    array_t = np.arange(1, 10, 1)
    E = np.zeros([nb, nc])
    for i in range(nb):
        for j in range(nc):
            E_ti = 0
            bi = array_b[i]
            ci = array_c[j]
            for ti in array_t:
                fbc = s4(bi, ci, ti)
                ft = s4(b, c, ti)
                E_ti += pow(ft - fbc, 2)
            E[i, j] = E_ti

    # Emin
    Emin = np.amin(E)
    pos_min = np.where(E == np.amin(E))
    i_min = pos_min[0][0]
    j_min = pos_min[1][0]

    # Emax
    Emax = np.amax(E)
    pos_max = np.where(E == np.amax(E))
    i_max = pos_max[0][0]
    j_max = pos_max[1][0]

    # Plot a 3D surface
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(array_b, array_c, E, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('b')
    ax.set_ylabel('c')
    ax.set_zlabel('E')
    ax.set_xlim(b - 1.2, b + 1.2)
    ax.set_ylim(c - 2.2, c + 2.2)
    ax.set_zlim(Emin, Emax)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # dots in surface
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(array_b, array_c, E, cmap='Purples_r')
    ax.set_xlabel('b')
    ax.set_ylabel('c')
    ax.set_zlabel('E')
    ax.set_zlim(Emin, Emax)
    ax.scatter3D(array_b[i_min], array_c[j_min], Emin, c='red', marker='o')
    ax.scatter3D(array_b[i_max], array_c[j_max], Emax, c='black', marker='o')
    # Edots
    for i in range(1, i_min):
        ax.scatter3D(array_b[i], array_c[i], E[i, i], s=20, c='black', marker='o')
    plt.title('Surface E')

    # Plot a 3D scontour

    fig = plt.figure()
    ax = fig.gca()
    ax.contour(array_b, array_c, E, 200, origin='lower', cmap='Purples_r', linewidths=2, extent=(-3, 3, -2, 1))
    plt.plot(array_b[i_min], array_c[j_min], c='red', marker='o')
    plt.plot(array_b[i_max], array_c[i_max], c='black', marker='o')
    for i in range(1, i_min):
        if (math.fmod(i, 4) == 0):
            plt.plot(array_b[i], array_c[i], c='black', marker='o')
    ax.set_xlim(b - 1.2, b + 1.2)
    ax.set_ylim(c - 2.2, c + 2.2)
    plt.title('Contour plot of surface E')

    # Print E (printE) with array_b in [2.7, 3.3], array_c in [4.4, 5.6]
    printE = np.zeros([10, 10])
    printE = E[10:23, 10:23]
    plt.figure()
    plt.plot(printE)
    plt.title('printE')


def main():
    s1 = get_row(const.a, const.b)
    s2 = get_row(const.a, const.b, const.c)
    s3 = get_row(const.a, const.b, const.c, const.d)
    # without_noize(s1, s2, s3)
    # with_noise(s1, s2, s3)
    exponential()


def get_noised_row(s):
    return [i + np.random.rand() - 0.5 for i in s]


def draw_row(s):
    plt.plot(s, const.arange)


if __name__ == '__main__':
    main()
