import math

import numpy as np
import matplotlib.pyplot as plt

init_centroids = np.random.rand(4, 2)
points = np.random.rand(100000, 2)

class1 = np.array([])
class2 = np.array([])
class3 = np.array([])
class4 = np.array([])


def draw_centroids(centroids):
    plt.plot(centroids[0][0], centroids[0][1], 'or')
    plt.plot(centroids[1][0], centroids[1][1], 'ob')
    plt.plot(centroids[2][0], centroids[2][1], 'oy')
    plt.plot(centroids[3][0], centroids[3][1], 'og')


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


def draw_classes(classes):
    p1, = plt.plot(classes[0][:, 0], classes[0][:, 1], '.r')
    p2, = plt.plot(classes[1][:, 0], classes[1][:, 1], '.b')
    p3, = plt.plot(classes[2][:, 0], classes[2][:, 1], '.y')
    p4, = plt.plot(classes[3][:, 0], classes[3][:, 1], '.g')
    plt.legend([p1, p2, p3, p4, ], ["Class 1", "Class 2", "Class 3", "Class 4"])


def main():
    global class1, class2, class3, class4
    print("Centroids:")
    print(init_centroids)
    draw_centroids(init_centroids)
    print("Random points:")
    print(points)
    draw_points(points)

    for p in points:
        nearest_centroid = get_nearest_centroid(p, init_centroids)
        if np.array_equal(nearest_centroid, init_centroids[0]):
            class1 = np.append(class1, p)
        elif np.array_equal(nearest_centroid, init_centroids[1]):
            class2 = np.append(class2, p)
        elif np.array_equal(nearest_centroid, init_centroids[2]):
            class3 = np.append(class3, p)
        else:
            class4 = np.append(class4, p)

    class1 = np.reshape(class1, (-1, 2))
    class2 = np.reshape(class2, (-1, 2))
    class3 = np.reshape(class3, (-1, 2))
    class4 = np.reshape(class4, (-1, 2))
    classes = np.array([class1, class2, class3, class4])

    draw_classes(classes)

    print("Class 1:")
    print(class1)
    print("Class 2:")
    print(class2)
    print("Class 3:")
    print(class3)
    print("Class 4:")
    print(class4)

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()


if __name__ == "__main__":
    main()
