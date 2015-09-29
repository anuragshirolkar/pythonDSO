__author__ = 'anuragshirolkar'

import random
import time

class model():
    data = []
    eta = 3.0
    lmbd = 0.00001
    d = 2
    weight = [0]*(d+1)
    alpha = []
    m = 0

    '''
    Reading data from the file and creating data array
    as well as initialization
    '''
    def read_data(self):
        datafile = open('../data/a1a.txt')
        mx_feature = 0
        lines = datafile.readlines()
        for line in lines:
            splt1 = line.split()[1:]
            features = [int(x.split(':')[0]) for x in splt1]
            mx_feature = max(mx_feature, max(features))
        mx_feature = 123
        self.d = mx_feature
        self.m = len(lines)
        self.alpha = [0]*self.m
        self.weight = [0]*(self.d+1)
        for line in lines:
            x = [0] * (self.d+1)
            x[self.d] = 1
            for feature in line.split()[1:]:
                x[int(feature.split(':')[0]) - 1] = float(feature.split(':')[1])
            y = int(line.split()[0])
            self.data.append((y, x))
        # print self.data
        # print self.m, self.d
        return

    '''
    iteration <==> 1 epoch
    '''
    def iterate(self):
        indices = []
        for i in range(self.m):
            for j in range(self.d+1):
                indices.append((i,j))
        random.shuffle(indices)
        for (i, j) in indices:
            # print i, j
            # print self.alpha[0]
            if j == self.d:
                # ignore this
                self.weight[j] += self.eta*(self.alpha[i]*self.data[i][1][j]/self.m)
            else:
                self.weight[j] -= self.eta/self.m*(self.lmbd*self.weight[j] - self.alpha[i]*self.data[i][1][j])
            if self.alpha[i] * self.data[i][0] >= 0:
                # if i == 0:
                #     print "here", self.eta / (self.m*(self.d+1)), self.weight[j]*self.data[i][1][j]/self.m, (self.data[i][0] - self.alpha[i]/2)/(self.m*(self.d+1)), self.weight[j]*self.data[i][1][j]/self.m
                self.alpha[i] += self.eta * ((self.data[i][0] - self.alpha[i]/2.0)*1.0/(self.m*(self.d+1)) - self.weight[j]*self.data[i][1][j]*1.0/self.m)
                # if i == 0:
                #     print self.alpha[i]
            else:
                self.alpha[i] += self.eta * (- self.weight[j]*self.data[i][1][j]/self.m)
                self.alpha[i] = 0
        '''
        for j in range(self.d+1):
            for i in range(self.m):
                if j == self.d+1:
                    # ignore this
                    self.weight[j] += self.eta*(self.alpha[i]*self.data[i][1][j]/self.m)
                else:
                    self.weight[j] -= self.eta*(self.lmbd*self.weight[j]/self.m - self.alpha[i]*self.data[i][1][j]/self.m)
                if self.alpha[i] * self.data[i][0] >= 0:
                    self.alpha[i] += self.eta * ((self.data[i][0] - self.alpha[i]/2)/(self.m*(self.d+1)) - self.weight[j]*self.data[i][1][j]/self.m)
                else:
                    print "continuing"
                    continue
                    self.alpha[i] += self.eta * (- self.weight[j]*self.data[i][1][j]/self.m)
                    self.alpha[i] = 0'''

    '''
    dot product of two vectors a and b
    '''
    def _dot_product(self, a, b):
        if len(a) != len(b):
            return 0
        product = 0
        for i in range(len(a)):
            product += a[i] * b[i]
        return product

    '''
    squared hinge loss function <=> max(0, 1 - <w.x>*y)
    '''
    def _squared_hinge(self, w, x, y):
        return max(0, 1 - y*(self._dot_product(w, x)))**2

    '''
    The P(w) function
    '''
    def risk(self):
        risk_val = 0
        for i in self.weight[:-1]:
            risk_val += self.lmbd*i*i/2
        for i in range(self.m):
            risk_val += self._squared_hinge(self.weight, self.data[i][1], self.data[i][0])/self.m
        return risk_val

    '''
    The f(w, alpha) function on page no 3.
    '''
    def f_val(self, w, a):
        ans = 0
        for i in range(self.m):
            for j in range(self.d+1):
                ans += self.lmbd*w[j]*w[j]/(2*self.m) - a[i]*a[i]*self.data[i][0]*self.data[i][0]/(4*self.m*(self.d+1)) + a[i]*self.data[i][0]/(self.m*(self.d+1)) - a[i]*w[j]*self.data[i][1][j]/self.m
        return ans

    def _maximizing_alpha(self, w):
        a1 = []
        for i in range(self.m):
            a1i = 2*(self.data[i][0] - self._dot_product(self.weight, self.data[i][1]))
            if self.data[i][0]*a1i < 0:
                a1i = 0
            a1.append(a1i)
        return a1

    def _minimizing_weight(self, a):
        w1 = []
        for j in range(self.d+1):
            w1.append(sum([(self.alpha[i]*self.data[i][1][j]) for i in range(self.m)])/(self.lmbd*self.m))
        return w1


    '''
    dual gap in convergence analysis on page no 5
    '''
    def dual_gap(self):
        # w1 = []
        # for j in range(self.d+1):
        #     w1.append(sum([(self.alpha[i]*self.data[i][1][j]) for i in range(self.m)])/(self.lmbd*self.m))
        # a1 = []
        # for i in range(self.m):
        #     a1i = 2*(self.data[i][0] - self._dot_product(self.weight, self.data[i][1]))
        #     if self.data[i][0]*a1i < 0:
        #         a1i = 0
        #     a1.append(a1i)
        # print self.weight
        # print w1
        print self.f_val(self.weight, self._maximizing_alpha(self.weight)) - self.f_val(self._minimizing_weight(self.alpha), self.alpha)

    def print_min_around(self):
        backup_w = self.weight[:]
        min_val = self.risk()
        for i in range(100):
            new_w = backup_w
            for j in range(len(new_w)):
                new_w[j] += (random.random()-0.5)/1000
            self.weight = new_w
            min_val = min(min_val, self.risk())
        self.weight = backup_w
        return min_val

    def risk_gradient(self):
        grad = []
        for j in range(self.d+1):
            val = self.lmbd*self.weight[j]
            for i in range(self.m):
                lin_loss = self.data[i][0]*self._dot_product(self.weight, self.data[i][1])
                # print lin_loss
                if lin_loss < 1:
                    val -= (1.0/self.m)*2*max(0, 1-lin_loss)*self.data[i][0]*self.data[i][1][j]
            grad.append(val)
        return grad



from math import sqrt
# main code

mdl = model()
mdl.read_data()
# print mdl.risk()
# print mdl.data[0][1]
risk = 1
for i in range(1000):
    #print mdl.data[0][0], mdl.alpha[0], mdl.risk()
    if i % 10 == 0:
        print mdl.risk()
    mdl.iterate()
    risk = min(risk, mdl.risk())
    #print mdl.alpha
print risk
# print mdl.weight
# print mdl.print_min_around()
# print mdl.risk()
# mdl.weight = [2.802292389673284, 7.809961395997025, -31.489365639256892]
# print mdl.risk()

def normal_gradient_descent():
    mdl = model()
    mdl.read_data()
    print mdl.risk()
    for i in range(100):
        if i % 1 == 0:
            print mdl.risk()
        grad = mdl.risk_gradient()
        # print grad
        for j in range(mdl.d+1):
            mdl.weight[j] -= mdl.eta*grad[j]/3

# normal_gradient_descent()
# min is less than 0.435069393306 0.425756661903 0.418955400011