from matplotlib import pyplot as plt
import numpy as np



def plot_decision_boundary(pred_func,X,model):
    #print X
    x1_min,x1_max = X[:,0].min() - .1,X[:,0].max()+.1
    x2_min,x2_max = X[:,1].min() - .1,X[:,1].max()+.1
    #print x1_min,x1_max
    y = X[:,2]
    h = 0.01
    #print np.linspace(x1_min,x1_max,h)
    xx,yy = np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x2_min,x2_max,h))
    print '$$$$$$$$$$$$$$$$$$$$$$$$'
    print xx.ravel(),yy.ravel()
    #z = pred_func(np.c_[xx.ravel(),yy.ravel()])
    z,v,b= pred_func(xx.ravel(), yy.ravel(),model)
    z = z.reshape(xx.shape)
    plt.contourf(xx,yy,z,cmap = plt.cm.Spectral)
    plt.scatter(X[:,0],X[:,1],c = y,cmap=plt.cm.Spectral)
    plt.show()
