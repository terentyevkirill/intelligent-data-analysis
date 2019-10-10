import math
import numpy as np
import matplotlib.pyplot as plt

points_num = 10000
clusters_num = 6
k = 0.0001
pause = 0.01

def distance(p1, p2):
    # print('p1:')
    # print(p1)
    # print('p2:')
    # print(p2)
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_nearest_centroid(p, centroids):
    distances = []
    for i in range(clusters_num):
        distances.append(distance(p, centroids[i]))
    # distances = [distance(p, c) for c in centroids]
    index = np.where(distances == np.amin(distances))
    return centroids[index]


def get_centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return [sum(x) / len(points), sum(y) / len(points)]


def draw_classes(classes):
    colors = ['r', 'b', 'y', 'g', 'c', 'm']
    for i in range(clusters_num):
        plt.plot(classes[i][:, 0], classes[i][:, 1], '.' + colors[i])


def is_end(prev_centroids, centroids):
    global clusters_num
    end = True
    for i in range(clusters_num):
        if distance(prev_centroids[i], centroids[i]) > k:
            end = False

    return end


def classify(points, centroids):
    classes = [[] for _ in range(clusters_num)]
    for p in points:
        nearest_centroid = get_nearest_centroid(p, centroids)
        nearest_centroid = np.reshape(nearest_centroid, (1, 2))
        for i in range(clusters_num):
            if np.array_equal(nearest_centroid[0], centroids[i]):
                classes[i] = np.append(classes[i], [p])

    for i in range(clusters_num):
        classes[i] = np.reshape(classes[i], (-1, 2))

    # print(classes)
    return classes


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
    global pause
    plt.pause(pause)
    plt.clf()
    axes = plt.gca()
    plt.grid(True)
    axes.set_xlim([-0.1, 1.1])
    axes.set_ylim([-0.1, 1.1])

    draw_classes(classes)
    # draw_centroids(centroids)
    plt.show()


def k_means(points, centroids):
    global clusters_num
    classes = classify(points, centroids)
    build_plot(classes)
    while True:
        prev_centroids = centroids
        centroids = [get_centroid(c) for c in classes]
        centroids = np.reshape(centroids, (clusters_num, 2))
        classes = classify(points, centroids)
        update_plot(classes)
        if is_end(prev_centroids, centroids):
            break

    print('END')
    plt.pause(10000)


def main():
    points = np.random.rand(points_num, 2)
    centroids = np.random.rand(clusters_num, 2)
    k_means(points, centroids)

    # plt.xlim((-0.1, 1.1))
    # plt.ylim((-0.1, 1.1))
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    main()
