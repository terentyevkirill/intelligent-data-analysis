import math

import numpy as np
import matplotlib.pyplot as plt

centroids = np.random.rand(4, 2)
points = np.random.rand(1000, 2)
k = 0.0001
cluster_num = 4

class1 = np.array([])
class2 = np.array([])
class3 = np.array([])
class4 = np.array([])

def draw_centroids(centroids):
    plt.plot(centroids[0][0], centroids[0][1], 'ok')
    plt.plot(centroids[1][0], centroids[1][1], 'ok')
    plt.plot(centroids[2][0], centroids[2][1], 'ok')
    plt.plot(centroids[3][0], centroids[3][1], 'ok')


def draw_points(points):
    plt.plot(points[:, 0], points[:, 1], '.k')


def distance(p1, p2): return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_nearest_centroid(p, centroids):
    d1 = distance(p, centroids[0])
    d2 = distance(p, centroids[1])
    d3 = distance(p, centroids[2])
    d4 = distance(p, centroids[3])
    d_min = min(d1, d2, d3, d4)
    if d_min == d1:
        return centroids[0]
    elif d_min == d2:
        return centroids[1]
    elif d_min == d3:
        return centroids[2]
    else:
        return centroids[3]


def get_centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return [sum(x) / len(points), sum(y) / len(points)]


def draw_classes(classes):
    plt.plot(classes[0][:, 0], classes[0][:, 1], '.r')
    plt.plot(classes[1][:, 0], classes[1][:, 1], '.b')
    plt.plot(classes[2][:, 0], classes[2][:, 1], '.y')
    plt.plot(classes[3][:, 0], classes[3][:, 1], '.g')


def classify():
    global class1, class2, class3, class4
    for p in points:
        nearest_centroid = get_nearest_centroid(p, centroids)
        if np.array_equal(nearest_centroid, centroids[0]):
            class1 = np.append(class1, p)
        elif np.array_equal(nearest_centroid, centroids[1]):
            class2 = np.append(class2, p)
        elif np.array_equal(nearest_centroid, centroids[2]):
            class3 = np.append(class3, p)
        else:
            class4 = np.append(class4, p)


def clear_classes():
    global class1, class2, class3, class4
    class1 = np.array([])
    class2 = np.array([])
    class3 = np.array([])
    class4 = np.array([])


def reshape_classes():
    global class1, class2, class3, class4
    class1 = np.reshape(class1, (-1, 2))
    class2 = np.reshape(class2, (-1, 2))
    class3 = np.reshape(class3, (-1, 2))
    class4 = np.reshape(class4, (-1, 2))


def is_end(prev_centroids, centroids):
    end = True
    for i in range(cluster_num):
        if distance(prev_centroids[i], centroids[i]) > k:
            end = False

    return end


def build_plot(classes):
    draw_classes(classes)
    # draw_centroids(centroids)
    axes = plt.gca()
    axes.set_xlim([-0.1, 1.1])
    axes.set_ylim([-0.1, 1.1])
    plt.grid(True)
    plt.ion()
    plt.show()


def update_plot(classes):
    plt.pause(0.2)
    plt.clf()
    axes = plt.gca()
    axes.set_xlim([-0.1, 1.1])
    axes.set_ylim([-0.1, 1.1])
    draw_classes(classes)
    # draw_centroids(centroids)
    plt.show()


def k_means(classes):
    global centroids, class1, class2, class3, class4
    while True:
        clear_classes()
        prev_centroids = centroids
        centroids = [get_centroid(c) for c in classes if len(c) > 0]
        centroids = np.reshape(centroids, (cluster_num, 2))
        classify()
        reshape_classes()
        classes = np.array([class1, class2, class3, class4])
        update_plot(classes)
        if is_end(prev_centroids, centroids):
            break

    print('END')
    plt.pause(10000)


def main():
    global class1, class2, class3, class4, centroids
    print('Init centroids:')
    print(centroids)
    classify()
    reshape_classes()
    classes = np.array([class1, class2, class3, class4])
    # print('Init classes:')
    # print(classes)
    build_plot(classes)
    k_means(classes)

if __name__ == "__main__":
    main()
