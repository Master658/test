from libsvm.python.svmutil import *


y,x = svm_read_problem('/home/fangsh/Downloads/libsvm/heart_scale')
model = svm_train(y[:200],x[:200],'-c 4')
p_label,p_acc,p_val = svm_predict(y[200:],x[200:],model)
print (p_label,p_acc,p_val)
