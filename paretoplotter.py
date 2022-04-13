import numpy as np
import designFns
from designFns import OWrite
from numpy import genfromtxt
import sys
target_threshold = 400

def main():
    data = genfromtxt('full_traindata.csv',skip_header=1)

    tindices = np.array(list(range(len(data))))
    tindices = tindices.reshape((len(tindices),1))
    data     = np.append(data,tindices,axis=1)

    data = data[~np.any(np.isnan(data), axis=1),:]

    data = data[(data[:,-2]<target_threshold) & (data[:,-3]>0.05)]

    data = data[:,-3:]
    idata = genfromtxt('full_train_ids.csv',dtype='str',skip_header=1)

    data = data.tolist()
    for i,item in enumerate(data):
	print(item)
        data[i] = data[i] + idata[int(round(item[-1]))].tolist()

    print(data)
    data = np.array(data)

    print(('Shape of final data: '+str(data.shape)))

    np.savetxt('full_PFdata.csv',data,delimiter=',',fmt='%s')


    ydata = -1.0*(data[:,:2])

    O_PF = designFns.pareto_front(ydata)
    for item in O_PF:
        print((data[int(round(item[2]))]))
    #designFns.plot_pareto(ydata,O_PF,label='overall',path='./')


main()
