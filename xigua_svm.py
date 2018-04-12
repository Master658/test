from libsvm.python.svmutil import *
import numpy as np 
from plot_decision_boundary import plot_decision_boundary

y, x = svm_read_problem('/home/fangsh/svm_test/xigua_svm_data')
model = svm_train(y[:16],x[:16],'-t 0')
#p_label, p_acc, p_vals = svm_predict(y[16:],x[16:],model)
#print p_label,p_acc,p_vals

from matplotlib import pyplot as plt
a = np.array([[0.697,0.460,1],
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
#x1 = a[:,0]
#x2 = a[:,1]
#y = a[:,2]
#plt.figure(figsize=(1,1),dpi=100)
#plt.scatter(x1,x2,s=100,c=y)
#plt.show()
plot_decision_boundary(svm_predict,a,model)
