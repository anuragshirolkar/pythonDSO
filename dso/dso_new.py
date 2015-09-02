__author__ = 'anuragshirolkar'

class model():
    data = []
    eta = 0.1
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
        print self.data
        print self.m, self.d
        return

    '''
    iteration <==> 1 epoch
    '''
    def iterate(self):
        for i in range(self.m):
            for j in range(self.d+1):
                if j == self.d:
                    # ignore this
                    self.weight[j] += self.eta*(self.alpha[i]*self.data[i][1][j]/self.m)
                else:
                    self.weight[j] -= self.eta*(self.lmbd*self.weight[j]/self.m - self.alpha[i]*self.data[i][1][j]/self.m)
                if self.alpha[i] * self.data[i][0] >= 0:
                    self.alpha[i] += self.eta * ((-self.data[i][0] + self.alpha[i])/(self.m*(self.d+1)) - self.weight[j]*self.data[i][1][j]/self.m)
                else:
                    self.alpha[i] += self.eta * (- self.weight[j]*self.data[i][1][j]/self.m)
                    self.alpha[i] = 0

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
        for i in self.weight:#[:-1]:
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
                ans += self.lmbd*w[j]*self.weight[j]/(2*self.m) - a[i]*self.alpha[i]*self.data[i][0]*self.data[i][0]/(4*self.m*(self.d+1)) + a[i]*self.data[i][0]/(self.m*(self.d+1)) - a[i]*w[j]*self.data[i][1][j]/self.m
        return ans

    '''
    dual gap in convergence analysis on page no 5
    '''
    def dual_gap(self):
        w1 = []
        for j in range(self.d+1):
            w1.append(sum([ (self.alpha[i]*self.data[i][1][j]) for i in range(self.m)])/(self.lmbd*self.m))
        a1 = []
        for i in range(self.m):
            a1.append(self.data[i][0] * max(0, 2*(1 - self._dot_product(self.weight, self.data[i][1]))))
        #print self.weight
        #print w1
        print self.f_val(self.weight, a1) , self.f_val(self.weight, self.alpha), self.f_val(w1, self.alpha)


'''
main code
'''
mdl = model()
mdl.read_data()
print mdl.risk()
print mdl.data[0][1]
risk = 1
for i in range(100):
    mdl.iterate()
    risk = min(risk, mdl.risk())
    #mdl.eta = mdl.eta/(i+2)*(i+1)
    if i% 10 == 0:
        print mdl.risk(), mdl.eta
        #mdl.dual_gap()
        #print mdl.weight
        #print mdl.alpha[:6]
    #print mdl.alpha
#print risk
#print mdl.weight