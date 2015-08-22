__author__ = 'anuragshirolkar'

from random import random

dimensions = 20
data_size = 1000
data = []

point1 = []
point2 = []

for i in range(dimensions) :
    point1.append(0)
    point2.append(10)

def random_point(point) :
    new_point = []
    for i in point :
        new_point.append(i + 10*random() - 5)
    return new_point

for i in range(data_size/2) :
    data.append((1, random_point(point1)))
    data.append((-1, random_point(point2)))

output_file = open('../data/input.txt', 'w')

for data_point in data :
    print >>output_file, data_point[0],
    for idx, feature in enumerate(data_point[1]) :
        print >>output_file, str(idx)+':'+str(feature),
    print >>output_file



