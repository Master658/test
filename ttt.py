from libsvm.python.svmutil import *
import numpy as np
from plot_decision_boundary import plot_decision_boundary
from matplotlib import pyplot as plt


y, x = svm_read_problem('/home/fangsh/svm_test/xigua_svm_data')
model = svm_train(y[:16],x[:16],'-t 0')
print '!!!!!!!!!!!!!!!!!!!!!!!!'
print x
print y
#p_label, p_acc, p_vals = svm_predict(y[16:],x[16:],model)
#print p_label,p_acc,p_vals


X = np.array([[0.697,0.460,1],
[0.774,0.376,1],
[0.634,0.264,1],
[0.608,0.318,1],
[0.556,0.215,1],
[0.403,0.237,1],
[0.481,0.149,1],
[0.437,0.211,1],
[0.666,0.091,-1],
[0.243,0.267,-1],
[0.245,0.057,-1],
[0.343,0.099,-1],
[0.639,0.161,-1],
[0.657,0.198,-1],
[0.360,0.370,-1],
[0.719,0.103,-1],
[0.593,0.042,-1]])

x1_min,x1_max = X[:,0].min() - .1,X[:,0].max()+.1
x2_min,x2_max = X[:,1].min() - .1,X[:,1].max()+.1
#print x1_min,x1_max
y2 = X[:,2]
h = 0.01
plt.figure(figsize=(1,1),dpi=100)
#print np.linspace(x1_min,x1_max,h)
xx,yy = np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x2_min,x2_max,h))
#print '$$$$$$$$$$$$$$$$$$$$$$$$'
#print xx.ravel(),yy.ravel()
#z = pred_func(np.c_[xx.ravel(),yy.ravel()])
test_x = []
test_y = []
for i in xrange(len(xx.ravel())):
    test_x.append({1:xx.ravel()[i],2:yy.ravel()[i]})
    test_y.append(+1)
z = svm_predict(test_y, test_x, model)[0]
z = z.reshape(xx.shape)
plt.contourf(xx,yy,z,cmap = plt.cm.Spectral)

plt.scatter(X[:,0],X[:,1],c = y2,s=100,cmap=plt.cm.Spectral)
plt.show()