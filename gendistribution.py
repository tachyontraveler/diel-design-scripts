import numpy as np
import finalnn
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pickle
from datetime import datetime
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import Normalizer, normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.model_selection import cross_val_predict, train_test_split
from multiprocessing import Pool
import finalnn
import sys
import os

n_btstrps = 200
random_state = 7
reuse_data = False
N_processors = 8
DATADIR = './datadir/cycle3/'
train_bg_min_cutoff   = 0.5
search_bg_min_cutoff = 5.0
def load_data(traindata_filename, 
        searchdata_filename, load_from_saved=False ):
    if load_from_saved:
        with open(os.path.join(DATADIR,load_from_saved),"rb") as fin:
            data = pickle.dump(fin)
            xt,yt,sdata = data[0], data[1], data[2]
        return (xt,yt,sdata)
    tdata = genfromtxt(traindata_filename, skip_header=1)
    tdata = tdata[~np.any(np.isnan(tdata), axis=1),:]
    tdata = tdata[(tdata[:,-1]>=train_bg_min_cutoff)]
    xt,yt = tdata[:,:-1], tdata[:,-1]
    yt    = np.log2(yt)
#    xt    = np.append(xt,tdata[:,-1].reshape((len(xt),1)),axis=1)
    print(('Shape of Xtrain : '+str(xt.shape)))
    print(('Shape of Ytrain : '+str(yt.shape)))

#    xt,xv,yt,yv = train_test_split(x,y,test_size=0.2,random_state=random_state)
    
    sdata = genfromtxt(searchdata_filename,skip_header=1)
    sindices = np.array(list(range(len(sdata))))
    sindices = sindices.reshape((len(sindices),1))
    sdata    = np.append(sdata,sindices,axis=1)

    sdata = sdata[~np.any(np.isnan(sdata), axis=1),:]
    sdata = sdata[(sdata[:,-2]>=search_bg_min_cutoff)]
    print(('Shape of Xsearch : '+str(sdata.shape)))
    sindices = sdata[:,-2:]
    sdata = sdata[:,:-1]
    scalar = StandardScaler()
    scalar.fit(xt)
    xt = scalar.transform(xt)
    sdata = scalar.transform(sdata)
    combined_features = FeatureUnion([("pca", PCA(n_components=150)),
                                      ('modelbased', SelectFromModel(estimator=RFR(n_estimators=470,
                                                                                   max_features=70),
                                                                    # threshold='mean',
                                                                    ))])
    xt = combined_features.fit(xt, yt).transform(xt)
    sdata = combined_features.transform(sdata)
    
    with open(os.path.join(DATADIR,"pickled_processed_data.pkl"),"wb") as fout:
        pickle.dump([xt,yt,sdata,sindices],fout)
    
    return (xt,yt,sdata)



def main():
    xt,yt,xs = load_data(
            traindata_filename  = os.path.join(DATADIR,"traindata.csv"),
            searchdata_filename = os.path.join(DATADIR,"searchdata.csv"),
            load_from_saved     = False
            )
    print('Shape of processed Xtrain : '+str(xt.shape), flush=True)
    print('Shape of processed Ytrain : '+str(yt.shape), flush=True)
    print(str(datetime.now())+' :: Processed Data', flush=True)
#    sample_weight = np.minimum(np.square(np.maximum(4.0*yt/target_threshold,1.0)),10.0)
#    sample_weight = np.minimum((np.maximum(4.0*yt/target_threshold,1.0)),4.0)

    pool = Pool(N_processors)

    design_args = []
    y_pred_distr=[]
    for it in range(n_btstrps):
        print('bootstrap number: '+str(it),flush=True)
        rbootstr = np.random.choice(len(xt),len(xt))
        x1 = np.array([xt[i] for i in rbootstr])
        y1 = np.array([yt[i] for i in rbootstr])
      # design_args.append( (x1,xs,y1) )
        y_pred_distr.append(helper_function((x1,xs,y1)))
        sys.stdout.flush()
#   y_pred_distr = pool.map(helper_function,design_args)
    with open(os.path.join(DATADIR, "ypred_distribution.pkl"),"wb") as fout:
        pickle.dump(y_pred_distr,fout)
    
def helper_function(args):
    return finalnn.predict(*args)


def predict_plot(y,ypred,path='./',idflag='ytest'):
    print((y.shape,ypred.shape))
    test_score = r2_score(y,ypred.flatten())
    spearman = spearmanr(y,ypred.flatten())[0]
    pearson = pearsonr(y,ypred.flatten())[0]
    print((test_score,spearman,pearson))
    plt.scatter(y,ypred,c='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    plt.xlabel('real values')
    plt.ylabel('fitted values')
    plt.title("Model validation")
    score_list = 'r2: '+str(round(test_score,2))+',  spearman:'+str(round(spearman,2))+',  pearson:'+str(round(pearson,2))
    plt.figtext(0.99, 0.01, score_list, horizontalalignment='right')
    figname = path+'plot_model_'+idflag
    plt.savefig(figname)
    plt.close()


main()
