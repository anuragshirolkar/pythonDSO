__author__ = 'anuragshirolkar'

from random import random

datafile = open('../data/twofeature.txt')

m = 0
d = 2


def process_line(line) :
    splitted = line.split()
    out = int(splitted[0])
    features = [0]*2 #[0]*(119)
    #features[2] = 1
    for feature in splitted[1:] :
        feature_split = feature.split(':')
        features[int(feature_split[0])-1] = float(feature_split[1])
    return (out, features)

data = []


for line in datafile.readlines():
    data.append(process_line(line))

a = [0] * len(data)
m = len(data)
d = len(data[0][1])
print d
w = [0]*d
eta = 0.01
lmbd = 0.0001

for t in range(10000):
    #print a
    if t % 1000 == 0:
        print t
    for i in range(m):
        for j in range(d):
            w[j] -= eta*( (lmbd*w[j]/m) - (a[i]*data[i][1][j]/m))
            if data[i][0]*a[i] >= 0:
                a[i] += eta*( - (data[i][0] - a[i]/2)/(m*d) - (w[j]*data[i][1][j]/m))
            else :
                a[i] += eta*(- (w[j]*data[i][1][j]/m) )


def dot_product(a, b) :
    if len(a) != len(b) :
        return 0
    product = 0
    for i in range(len(a)) :
        product += a[i] * b[i]
    return product


def squared_hinge(weight, x, y) :
    return max(0, 1 - y*(dot_product(weight, x)))**2


def risk(weight):
    risk_val = 0
    for i in weight:
        risk_val += lmbd*i*i/2
    for i in range(m) :
        risk_val += squared_hinge(weight, data[i][1], data[i][0])/m
    return risk_val

min_val = 1000

for i in range(100) :
    weight = []
    for j in range(d) :
        weight.append(w[j] + random()/1000 - 0.0005)
    if risk(weight) < min_val :
        min_val = risk(weight)


print risk(w), min_val
if risk(w) <= min_val :
    print 'hurray'
print w

