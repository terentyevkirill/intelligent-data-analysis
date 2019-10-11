import math
import numpy as np
import matplotlib.pyplot as plt
import sys

points_num = 0  # число точек, подлежащих класеризации
clusters_num = 0    # число кластеров
k = 0.0001          # точность алгоритма кластеризации
pause = 0.001       # пауза между отрисовками итераций


def distance(p1, p2):
    '''
    Получение эвклидового расстояния между двумя точками
    '''
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_nearest_centroid(p, centroids):
    '''
    Получение ближайшего центроида к данной точке
    :param p: точка
    :param centroids: массив центроидов (число = числу кластеров)
    :return: координаты ближайшего центроида к точке p
    '''
    distances = []
    for i in range(clusters_num):
        distances.append(distance(p, centroids[i]))
    # distances = [distance(p, c) for c in centroids]
    index = np.where(distances == np.amin(distances))
    return centroids[index]


def get_centroid(points):
    '''
    Нахождение центроида для массива точек
    :param points: массив точек
    :return: координаты центроида
    '''
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return [sum(x) / len(points), sum(y) / len(points)]


def draw_classes(classes):
    ''' Отрисовка классов '''
    colors = ['r', 'b', 'y', 'g', 'c', 'm']
    for i in range(clusters_num):
        plt.plot(classes[i][:, 0], classes[i][:, 1], '.' + colors[i])


def draw_centroids(centroids):
    ''' Отрисовка центроидов '''
    for c in centroids:
        plt.plot(c[0], c[1], 'ok')


def is_end(prev_centroids, centroids):
    '''
    Проверка на завершение алгоритма
    :param prev_centroids: центроиды классов на предыдущей итерации
    :param centroids: центроиды классов на текущей итерации
    :return: завершился ли алгоритм
    '''
    global clusters_num
    end = True
    for i in range(clusters_num):
        if distance(prev_centroids[i], centroids[i]) > k:
            end = False

    return end


def classify(points, centroids):
    '''
    Классификация массива точек по кластерам (ближайший центроид к точке)
    :param points: массив точек подлежащих кластеризации
    :param centroids: массив центроидов
    :return: массив классов (класс = массив точек)
    '''
    classes = [[] for _ in range(clusters_num)]
    for p in points:
        nearest_centroid = get_nearest_centroid(p, centroids)
        nearest_centroid = np.reshape(nearest_centroid, (1, 2))
        for i in range(clusters_num):
            if np.array_equal(nearest_centroid[0], centroids[i]):
                classes[i] = np.append(classes[i], [p])

    for i in range(clusters_num):
        classes[i] = np.reshape(classes[i], (-1, 2))

    return classes


def build_plot(classes, centroids):
    ''' Настройка графика (установка) '''
    draw_classes(classes)
    draw_centroids(centroids)
    axes = plt.gca()
    axes.set_xlim([-0.1, 1.1])
    axes.set_ylim([-0.1, 1.1])
    plt.text(0, 1.15, 'Iteration 1')
    plt.grid(True)
    plt.ion()
    plt.show()


def update_plot(classes, centroids, i):
    ''' Перерисовка графика (цикл) '''
    global pause
    plt.pause(pause)
    plt.clf()
    axes = plt.gca()
    plt.grid(True)
    axes.set_xlim([-0.1, 1.1])
    axes.set_ylim([-0.1, 1.1])
    plt.text(0, 1.15, 'Iteration ' + str(i))
    draw_classes(classes)
    draw_centroids(centroids)
    plt.show()


def k_means(points, centroids):
    '''
    Алгоритм кластеризации
    :param points: точки для кластеризации
    :param centroids: массив центроидов (изначальный)
    '''
    global clusters_num
    classes = classify(points, centroids)
    build_plot(classes, centroids)
    i = 1
    while True:
        i += 1
        prev_centroids = centroids
        centroids = [get_centroid(c) for c in classes]
        centroids = np.reshape(centroids, (clusters_num, 2))
        classes = classify(points, centroids)
        update_plot(classes, centroids, i)
        if is_end(prev_centroids, centroids):
            plt.text(0.4, 1.15, 'Converged!')
            plt.show()
            break

    print('END')
    plt.pause(10000)


def main():
    global points_num, clusters_num
    points_num = int(sys.argv[1])
    clusters_num = int(sys.argv[2])

    points = np.random.rand(points_num, 2)
    centroids = np.random.rand(clusters_num, 2)
    k_means(points, centroids)


if __name__ == "__main__":
    main()
