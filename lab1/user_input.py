import numpy as np
import matplotlib.pyplot as plt

# вариант 3: 1,2,5,6 класс; между объектами 2,3 -
# расстояние минковского, сумма модулей разностей значений каждого признака;
# между объектом и классом 1,4 -
# Расстояние до центроиду класса (эталона, усредненных по каждому признаку)
# наибольшее из значений расстояния до всех эталонов класса ( «дальний сосед»)

# ручной ввод точки

class1 = np.array(
    [[0.05, 0.91],
     [0.14, 0.96],
     [0.16, 0.9],
     [0.07, 0.7],
     [0.2, 0.63]])
class2 = np.array(
    [[0.49, 0.89],
     [0.34, 0.81],
     [0.36, 0.67],
     [0.47, 0.49],
     [0.52, 0.53]])
class5 = np.array(
    [[0.31, 0.43],
     [0.45, 0.27],
     [0.33, 0.16],
     [0.56, 0.29],
     [0.54, 0.13]])
class6 = np.array(
    [[0.05, 0.15],
     [0.09, 0.39],
     [0.13, 0.51],
     [0.25, 0.34],
     [0.15, 0.36]])

def centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return [sum(x) / len(points), sum(y) / len(points)]


def minkovsky_dist(o1, o2, k):
    if not isinstance(k, int) or k < 2:
        raise ValueError("k should be integer, k>2")
    dx = abs(o1[0] - o2[0]) ** k
    dy = abs(o1[1] - o2[1]) ** k
    return (dx + dy) ** (1 / k)


def sumAbs_dist(o1, o2):
    return abs(o1[0] - o2[0]) + abs(o1[1] - o2[1])

# пользовательский ввод координат рспознаваемого объекта
while True:
    print("Введите признаки объекта (координаты точки [0, 1]):")
    x = float(input("x: "))
    y = float(input("y: "))
    if abs(x) > 1 or abs(y) > 1:
        continue
    else:
        break

x = float(x)
y = float(y)
a = [x, y]
print("Признаки объекта: ")
print(a)

# пользовательский ввод методов
while True:
    print("Выберите метод вычисления расстрояния между двумя объектами в двумерном пространстве:")
    print("1. Расстрояние Минковского")
    print("2. Сумма модулей разностей значений каждого признака")
    objToObjMethod = int(input())
    if objToObjMethod not in [1, 2]:
        continue
    else:
        break

while True:
    print("Выберите метод вычисления расстрояния между объектом и классом:")
    print("1. Расстояние до центроида класса")
    print("2. Наибольшее из значений расстояния до всех эталонов класса")
    objToClassMethod = int(input())
    if objToClassMethod not in [1, 2]:
        continue
    else:
        break


def funObjToObj(function, a, e, *args):
    distances.append(function(a, e, *args))


def funObjToClass(function, a, *args):
    distancesToEtalone = []
    points = []
    for c in [class1, class2, class5, class6]:
        for e in c:
            distancesToEtalone.append(function(a, e, *args))
        max_index = np.where(distancesToEtalone == np.max(distancesToEtalone))[0][0]
        points.append(c[max_index])
        # print(distancesToEtalone)
        distancesToEtalone.clear()
    for p in points:
        distances.append(function(a, p, *args))
    print(distances)

# программа
distances = []
if objToObjMethod == 1:
    while True:
        print("Введите коэффициент для расстояния Минковского (целое число > 2):")
        minkovsky_k = int(input())
        if not isinstance(minkovsky_k, int) or minkovsky_k < 2:
            continue
        else:
            break

    if objToClassMethod == 1:
        for c in [class1, class2, class5, class6]:
            funObjToObj(minkovsky_dist, a, centroid(c), minkovsky_k)
    else:
        funObjToClass(minkovsky_dist, a, minkovsky_k)
else:
    if objToClassMethod == 1:
        for c in [class1, class2, class5, class6]:
            funObjToObj(sumAbs_dist, a, centroid(c))

    else:
        funObjToClass(sumAbs_dist, a)

# окраска введенного объекта в цвета класса, к которому он принадлежит
min_index = np.where(distances == np.min(distances))[0][0]
if min_index == 0:
    print("Объект относится к классу 1")
    p0, = plt.plot(a[0], a[1], '*r')
elif min_index == 1:
    print("Объект относится к классу 2")
    p0, = plt.plot(a[0], a[1], '*b')
elif min_index == 2:
    print("Объект относится к классу 5")
    p0, = plt.plot(a[0], a[1], '*c')
else:
    print("Объект относится к классу 6")
    p0, = plt.plot(a[0], a[1], '*g')

for i in enumerate(class1):
    p1, = plt.plot(class1[:, 0], class1[:, 1], 'sr')

for i in enumerate(class2):
    p2, = plt.plot(class2[:, 0], class2[:, 1], 'Db')

for i in enumerate(class5):
    p5, = plt.plot(class5[:, 0], class5[:, 1], 'oc')

for i in enumerate(class6):
    p6, = plt.plot(class6[:, 0], class6[:, 1], '^g')

plt.legend([p1, p2, p5, p6, p0], ["class1", "class2", "class5", "class6", "object"])
plt.xlabel("x")
plt.ylabel("y")
plt.show()
