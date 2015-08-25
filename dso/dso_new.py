__author__ = 'anuragshirolkar'

class model():
    data = []
    eta = 0.1
    lmbd = 0.00001
    d = 2
    weight = [0]*(d+1)
    alpha = []
    m = 0

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
        self.weight = [0.1]*(self.d+1)
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

    def iterate(self):
        for i in range(self.m):
            for j in range(self.d+1):
                if j == self.d:
                    self.weight[j] += self.eta*(self.alpha[i]*self.data[i][1][j]/self.m)
                else:
                    self.weight[j] -= self.eta*(self.lmbd*self.weight[j]/self.m - self.alpha[i]*self.data[i][1][j]/self.m)
                if self.alpha[i] * self.data[i][0] >= 0:
                    self.alpha[i] += self.eta * ((-self.data[i][0] + self.alpha[i])/(self.m*(self.d+1)) - self.weight[j]*self.data[i][1][j]/self.m)
                else:
                    self.alpha[i] += self.eta * (- self.weight[j]*self.data[i][1][j]/self.m)
                    self.alpha[i] = 0

    def _dot_product(self, a, b):
        if len(a) != len(b):
            return 0
        product = 0
        for i in range(len(a)):
            product += a[i] * b[i]
        return product

    def _squared_hinge(self, w, x, y):
        return max(0, 1 - y*(self._dot_product(w, x)))**2

    def risk(self):
        risk_val = 0
        for i in self.weight[:-1]:
            risk_val += self.lmbd*i*i/2
        for i in range(self.m):
            risk_val += self._squared_hinge(self.weight, self.data[i][1], self.data[i][0])/self.m
        return risk_val

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
        #print mdl.weight
        #print mdl.alpha[:6]
    #print mdl.alpha
print risk
print mdl.weight
# [0.006887633299861208, 0.014764132297201192, 0.01240469014529335]
# [0.009355598828895353, 0.01911519415644527, 0.02221181289762343]
# [0.008866639238918294, 0.013442128550192464, 0.028414838014948193]
# [-0.002104333821542469, -0.010286502405179206, 0.03340979119748148]