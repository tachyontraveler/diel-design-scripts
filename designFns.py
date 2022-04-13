import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import euclidean
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import bottleneck
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import cross_val_score,KFold,cross_val_predict
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor as RFR
from random import shuffle
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import make_scorer
##|------------------------------------------------------------------------
def dominate(x1,x2):        # To check dominance between two points
    return (x1 >= x2).all()
    
def g_cdf(a,b,s):           # Gaussian cumulative distribution function
    return (norm.cdf(np.divide((a-b),s)))

def g_pdf(a,b,s):           # Gaussian probability distribution function
    return (norm.pdf(np.divide((a-b),s)))

##|------------------------------------------------------------------------
def spearscore(y1,y2):
    a = np.array([[y1[i],y2[i]] for i in range(len(y1))])
    return np.mean(spearmanr(a,axis=0)[0])
##|---------------------------------------------------------------------------------------------------------###

SpearmanRSc = make_scorer(spearscore)
kfcv = KFold(n_splits=3,shuffle=True)
pca_n_components_parm = [60]
univariate_score_func_parm = [f_regression]
univariate_k_parm = [100]

##|------------------------------------------------------------------------
#   < Function to calculate Pareto Front of a multiobjective data array >
#   < By default, the paretofront function minimizes the objectives >
def pareto_front(MO):
    PF = np.copy(MO)
    indices = np.arange(0,len(MO))
    indices.shape=(len(MO),1)
    PF = np.append(PF,indices,axis=1)
    for i in range(len(MO)):
        to_delete=[]
        for j in range(len(PF)):
            if i!=PF[j][2] and dominate(PF[j][:2],MO[i]):  to_delete.append(j)
        PF = np.delete(PF,to_delete,0)
    OWrite('created pareto front of size: '+str(len(PF)))
    PF = PF[PF[:,0].argsort()]
    return (PF) 

##|----------------------------------------------------------------------------------------------------------------------------------------
#   < Function to calculate P(Improvement) for a point from sub-PF >
#   < Inputs : (mu,sigma) of a candidate point, Sub-PF
def probability_improvement(PF,mu,sigma):   
    PI,temp_sum=0.0,0.0
    PI=PI+g_cdf(PF[0][0],mu[0],sigma[0])
    for i in range(len(PF)-1):
        temp_sum = temp_sum + ((g_cdf(PF[i+1][0],mu[0],sigma[0]) - g_cdf(PF[i][0],mu[0],sigma[0])) * g_cdf(PF[i+1][1],mu[1],sigma[1]))
    PI=PI+temp_sum
    PI=PI+((1.0-g_cdf(PF[-1][0],mu[0],sigma[0])) * g_cdf(PF[-1][1],mu[1],sigma[1]))
    return (PI)

##|-----------------------------------------------------------------------------------------------------------------------------------------
#   < Function to find centroid of predictive distribution in region of improvement >
#   < Inputs: Sub-PF, P(Improvement), (mu,sigma) of a candidate point
def centroid(PF,mu,sigma,PI):
    Y1,temp_sum = 0.0,0.0
    Y1 = Y1 + (mu[0]*g_cdf(PF[0][0],mu[0],sigma[0])) - (sigma[0]*g_pdf(PF[0][0],mu[0],sigma[0]))
    for i in range(len(PF)-1):
        t1 = (mu[0]*g_cdf(PF[i+1][0],mu[0],sigma[0])) - (sigma[0]*g_pdf(PF[i+1][0],mu[0],sigma[0]))
        t2 = (mu[0]*g_cdf(PF[i][0],mu[0],sigma[0])) - (sigma[0]*g_pdf(PF[i][0],mu[0],sigma[0]))
        temp_sum = temp_sum + ((t1-t2) * g_cdf(PF[i+1][1],mu[1],sigma[1]))
    Y1 = Y1 + temp_sum
    Y1 = Y1 + ( ((mu[0]*(1-g_cdf(PF[-1][0],mu[0],sigma[0]))) + (sigma[0]*g_pdf(PF[-1][0],mu[0],sigma[0]))) * g_cdf(PF[-1][1],mu[1],sigma[1]))
    Y1 = np.divide(Y1,PI)

    Y2,temp_sum = 0.0,0.0
    Y2 = Y2 + (mu[1]*g_cdf(PF[-1][1],mu[1],sigma[1])) - (sigma[1]*g_pdf(PF[-1][1],mu[1],sigma[1]))
    for i in range(len(PF)-1):
        t1 = (mu[1]*g_cdf(PF[i][1],mu[1],sigma[1])) - (sigma[1]*g_pdf(PF[i][1],mu[1],sigma[1]))
        t2 = (mu[1]*g_cdf(PF[i+1][1],mu[1],sigma[1])) - (sigma[1]*g_pdf(PF[i+1][1],mu[1],sigma[1]))
        temp_sum = temp_sum + ((t1-t2) * g_cdf(PF[i][0],mu[0],sigma[0]))
    Y2 = Y2 + temp_sum
    Y2 = Y2 + ( ((mu[1]*(1-g_cdf(PF[0][1],mu[1],sigma[1]))) + (sigma[1]*g_pdf(PF[0][1],mu[1],sigma[1]))) * g_cdf(PF[0][0],mu[0],sigma[0]))
    Y2 = np.divide(Y2,PI)

    return(np.array([Y1,Y2]))

##|-----------------------------------------------------------------------------------------------------------------------------------------
#   < TO find Minimum distance between two points in Euclidean space >
#   < Used to find Improvement in Centroid based design  >
def minimum_distance(PF,C):
    return min([euclidean(x[:2],C) for x in PF])

##|--------------------------------------------------------------------
#   < To find Minimum distance of a point from Sub-PF  >
#   < Used to find Improvement in MaxiMin based design  >
def minimum_distancemm(PF,M):
    md = max([-min([M[i]-x[i] for i in range(len(M))]) for x in PF])
    if md > 0: return md
    else: return 0

##|-------------------------------------------------------------------
#   < To calculate Expected Improvement - Centroid based  >
def expected_improvementC(PF,C,PI):
    EI = PI * minimum_distance(PF,C)
    return EI

##|-------------------------------------------------------------------
#   < To calculate Expected Improvement - Maximin based  >
def expected_improvementMM(PF,M,PI):
    EI = PI * minimum_distancemm(PF,M)
    return EI



def expected_improvementSOO(hm,mu,sigma):
    ei = (sigma*g_pdf(mu,hm,sigma))+((mu-hm)*g_cdf(mu,hm,sigma))
    return ei



##|-----------------------------------------------------------------------------------
#   < Plot Pareto front of a data >
#   < Inputs: data(=y), PF, mode (='save' or 'show'),path to save the plotted image >
def plot_pareto(y,PF,y_selected=[],mode='save',path='./',label='current'):
    s=121
    plt.scatter(y[:,0], y[:,1], c='0.75',s=s/3, alpha=0.2,label='Whole data')
    plt.scatter(PF[:,0],PF[:,1],c='red', s=s/3,marker='s', label='Current Pareto Front')
    if len(y_selected)>0:
        y_selected=np.array(y_selected)
        plt.scatter(y_selected[:,0], y_selected[:,1],marker='^',c='green',s=s/2,label='Selected data')
    plt.xlabel('Property1')
    plt.ylabel('Property2')
    plt.title("Project title")
    pareto_dir = path if path[-1]=='/' else path+'/'
    if mode=='save': plt.savefig(pareto_dir+'pf_'+label)
    else: plt.show()
    plt.close()


##|-----------------------------------------------------------------------------------------
#   < To calculate mean and sigma of ditributions from a single regressor model itslef >
#   < Eg: Gaussian regressor. Use pickled regressors generated from SkLearn module only >
def model_distribution(x,y,search_X):
    pred_vals = []
    mlmodel = pickle.load(open('best_estimator','rb'))
    mlmodel = mlmodel.fit(x,y)
    means,sigmas = mlmodel.predict(search_X,return_std=True)
    return (np.array(means), np.array(sigmas))

##|------------------------------------------------------------------------------------------
#   < Returns an array containing indices of data points with largest expected improvement >
#   < Input: Array of Expected Improvement values, No. of datapoints needed for measurement >
#   < The returned array is NOT sorted internally in order of EI values >
#   < Efficient filtering algorithm for cases with >1 measurements per design cycle >
def measurement_selector(EIs,N_MEASUREMENTS=1):
    return ( np.array(bottleneck.argpartition(EIs,EIs.size-N_MEASUREMENTS)[-N_MEASUREMENTS:]) )


##|------------------------------------------------------------------------------------------
#   < Function to write output to output file 'outfile' and Terminal console >
#   < Timestamp is added to the output >
def OWrite(s,filename='log.design.out',pr=True,wr=True):
    s = '\n'+str(datetime.now())+' ::  '+s
    if pr:  print(s)
    if wr:
        with open(filename,'a') as fout:  fout.write(s)
##|-------------------------------------------------------------------------------------------

def bootstrap_validation(Xt,Yt,Xs,n_regress=100,n_bootstr=10):
    #pred_mean,pred_sigma =[0,0],[0,0]
    divs = np.maximum(np.absolute(np.mean(Xt,0)),1.0)
    Xt,Xs = Xt/divs, Xs/divs
    for iobj in range(1):#(Yt.shape[1]):
        pred_vals=[]
        OWrite('Considering the objective '+str(iobj))
        xt,yt,xs = np.copy(Xt), np.copy(Yt), np.copy(Xs)
        print(xt)
        print(yt)
        print(xs)
        sample_weight = np.square(np.maximum(2.0*yt/(np.amax(yt)),1.0))
        fit_params={'model__sample_weight':sample_weight}
        best_pipeline  = grid_search_rfr(xt,yt,fit_params=fit_params,obj_id=iobj)
        best_estimator = clone(best_pipeline.named_steps['model'])
        best_preproc   = clone(best_pipeline.named_steps['features'])
        xt = best_preproc.fit(xt,yt).transform(xt)
        xs = best_preproc.transform(xs)
        for ibootstr in range(n_bootstr*n_regress):
            rbootstr = np.random.choice(len(xt),len(xt))
            x1 = np.array([xt[i] for i in rbootstr])
            y1 = np.array([yt[i] for i in rbootstr])
            sw1 = np.array([sample_weight[i] for i in rbootstr])
            if ibootstr%n_bootstr == 0:
                mlmodel = clone(best_estimator)
            pred_vals.append(mlmodel.fit(x1,y1,sample_weight=sw1).predict(xs))
        pred_vals = np.array(pred_vals)
        OWrite('Finished the boostrapping')
        pred_mean = np.mean(pred_vals,axis=0)
        pred_sigma = np.std(pred_vals,axis=0)
    return(np.array(pred_mean),np.array(pred_sigma))


def prior_select(X,Y,n_train):
    x,y = ([] for i in range(2))
    # ovr_PF = pareto_front(np.array(Y))
    pareto_indices = Y.argsort()[-10:]
    selection_range = np.delete(np.arange(len(X)),pareto_indices,0)

    rselect = np.random.choice(selection_range,n_train,replace=False)
    x = np.array([X[rindex] for rindex in rselect])
    y = np.array([Y[rindex] for rindex in rselect])
    search_X = np.delete(X,rselect,0)
    search_Y = np.delete(Y,rselect,0)
    OWrite('Created prior with sampling size: '+str(n_train))
    return (x,y,search_X,search_Y)


def grid_search_svr(x1,y1,fit_params=None):
    pipeline = Pipeline([('features',FeatureUnion([('pca',PCA()),\
                              ('univariate',SelectKBest())])),   ('model',SVR()) ])
    print("Finding the best regressor model through grid search over parameters")
    model_C_parm = parm_span(2,0.0,8.0,1.0)  #prop. to Curvature of decision surfaces
    model_gamma_parm = parm_span(2,0.0,5.0,1.0) # Range of influence of a single data point (inverse proportion)
    model_epsilon_parm = [0.1]               # Error tolerance term

    parameters_svm = {'model__C':model_C_parm,
                      'model__epsilon':model_epsilon_parm,
                      'model__gamma':model_gamma_parm,
                      'features__pca__n_components':pca_n_components_parm,
                      'features__univariate__score_func':univariate_score_func_parm,
                      'features__univariate__k':univariate_k_parm}

    grid = GSCV(pipeline,param_grid=parameters_svm,cv=kfcv,n_jobs=10,scoring=SpearmanRSc)
    if fit_params: OWrite('Weights applied to samples')
    else: OWrite('No weights applied to samples')
    grid.fit(x1,y1,**fit_params)
    OWrite("Best estimator params : "+str(grid.best_params_))
    OWrite("Best estimator score : "+str(grid.best_score_))
#    pickle.dump(grid.best_estimator_,open('best_estimator','wb'))
    return (grid.best_estimator_)

def grid_search_rfr(x1,y1,fit_params=None,obj_id=0):
    pipeline = Pipeline([('features',FeatureUnion([('pca',PCA()),\
                              ('univariate',SelectKBest())])),   ('model',RFR()) ])
    print("Finding the best regressor model through grid search over parameters")
    model_n_estimators_parm = [60] #np.arange(50,71,10)#[int(item) for item in parm_span(10,0,2,0.05)] # Number of estimators (higher is better)
    model_max_features_parm = [np.arange(20,50,3),np.arange(25,50,3)]  # Number of features for best split. Should be 0< and < n_features
    model_n_jobs_parm = [-1]
    parameters_svm = {'model__n_estimators':model_n_estimators_parm,
                      'model__max_features':model_max_features_parm[obj_id],
                      'model__n_jobs':model_n_jobs_parm,
                      'features__pca__n_components':pca_n_components_parm,
                      'features__univariate__score_func':univariate_score_func_parm,
                      'features__univariate__k':univariate_k_parm}

    grid = GSCV(pipeline,param_grid=parameters_svm,cv=kfcv,n_jobs=-1)#,scoring=SpearmanRSc)
    grid.fit(x1,y1,**fit_params)
    OWrite("Best estimator params : "+str(grid.best_params_))
    OWrite("Best estimator score : "+str(grid.best_score_))
#    pickle.dump(grid.best_estimator_,open('best_estimator','wb'))
    return (grid.best_estimator_)


def parm_span(e,i,f,p):
    return([pow(e,x) for x in np.arange(i,f,p)])

