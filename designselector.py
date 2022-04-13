import pickle
import numpy as np
from numpy import genfromtxt
import designFns

DATADIR   = './datadir/cycle3/'
with open(os.path.join(DATADIR,'ypred_distribution.pkl'),"rb") as fin:
    ypreds   = np.array(pickle.load(fout))
with open(os.path.join(DATADIR,'pickled_processed_data.pkl'),"rb") as fin:
    sindices = pickle.load(fin)[-1]
hm       = np.log2(100) 
#hm      = np.amax(pickle.load(open('pickled_processed_data.pkl','r'))[2])
mu       = np.mean(ypreds,0).flatten()
sigma    = np.std(ypreds,0).flatten()
labels   = np.genfromtxt('./datasets/cycle2/new_search_identifiers.csv',dtype='str')
#header='material_id,compound,filename,delta_e,stability') 
EI = [round(designFns.expected_improvementSOO(hm,mu[i],sigma[i]),8) for i in range(len(mu))]

print(('The value of hm being used is : '+str(hm)))
np.savetxt(DATADIR+'EI_data.csv',[[sindices[i][-1],mu[i],sigma[i],EI[i]] for i in range(len(mu))])


print('\n\nsdata_index, mu, sigma, EI, material_id, compound, filename, delta_e, stability, bandgap')
print('\n\n-----Best promising 10 candidates based on EGO are:\n')
for i in np.array(EI).argsort()[-10:]:
    comp_info = [int(sindices[i][-1]),round(mu[i],4),round(sigma[i],4),round(EI[i],4)]
    for item in labels[int(sindices[i][-1])]: comp_info.append(item)
    comp_info.append(round(sindices[i][0],4))
    print(comp_info)

print('\n\n-----Best 10 candidates based on exploitation are:\n')
for i in np.array(mu).argsort()[-10:]:
    comp_info = [int(sindices[i][-1]),round(mu[i],4),round(sigma[i],4),round(EI[i],4)]
    for item in labels[int(sindices[i][-1])]: comp_info.append(item)
    comp_info.append(round(sindices[i][0],4))
    print(comp_info)
