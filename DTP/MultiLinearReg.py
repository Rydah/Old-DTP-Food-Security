import pandas as pd
import numpy as np
import random
from dataclasses import dataclass

@dataclass
class PredResult:
    target:str
    tfname:str #friendly name
    res:float #float value    


@dataclass
class NormalizationData():
    label:str
    vmean:float
    vstd:float
    vmin:float
    vmax:float
    mode:str = "Z"
    ohe_p:str= ""
    ohe:bool = False

class MlrResults:
    def __init__(self,features,feature_labels,tgtlabel,betas,J_hist,nrows,r2,mse,iters,alpha,split,rs,nd):
        self.f = features
        self.fl = feature_labels
        self.tl = tgtlabel
        self.b = betas
        self.Jhist = J_hist
        self.n = nrows
        self.r2 = r2
        self.mse = mse
        self.iters = iters
        self.alpha = alpha
        self.split = split
        self.rs = rs
        self.nd = nd
    def __str__(self):
        return f"{self.tl}:\n   Features: {len(self.f)}\n   Rows: {self.n}\n   Split: {self.split}\n   R2: {self.r2}\n   MSE: {self.mse}\n   Iters: {self.iters}\n   LRate: {self.alpha}\n"
    def describe_features(self):
        return str(self.fl)
        

class MultiLinear:
    def __init__(self,dataframe,norm_default="Z"):
        self.df = dataframe
        self.target = []
        self.features = []
        self._tgtset = None
        self._featureset = None
        self.NormDefault = norm_default
        self._normdata = {}
    @property
    def NormDefault(self):
        return self._defaultnorm
    @NormDefault.setter
    def NormDefault(self,n_default):
        if n_default == "MM" or n_default == "Z":
            self._defaultnorm = n_default
        else:
            raise Exception("Only 'Z' (Z Norm) or 'MM' (Min Max) modes are supported")
        
    @property
    def target(self):
        return self.__target
    @target.setter
    def target(self,targets):
        if not targets:
            self.__target = []
        if isinstance(targets,list):
            self.__target = targets
        elif isinstance(targets,str):
            self.__target = [targets]
        else:
            raise Exception("Not Supported")
    @property
    def features(self):
        return self.__features
    @features.setter
    def features(self,features):
        if not features:
            self.__features = []
        if isinstance(features,list):
            self.__features = features
        elif isinstance(features,str):
            self.__features = [features]
        else:
            raise Exception("Not Supported")

    def get_objt_cols(self,dfin):
        idx = 0
        _ = []
        l = list(dfin.columns)
        for t in list(dfin.dtypes):
            if t == "object":
                _.append(l[idx])
            idx+=1
        return _
            
    def _categorize_strs_ohe(self,dfin,objlist):#do after features and targets are defined
        if objlist:
            return pd.get_dummies(dfin,columns=objlist,dtype=int)
        else:
            return dfin.copy()
        
    def _normalize_z(self,dfin,skip=[]):
        dfout = dfin.copy()
        if skip:
            _activecol = [col for col in dfin.columns if not col in skip]
            dfs = dfout.filter(items=_activecol)
            dfsm = dfs.mean()
            dfss = dfs.std()
            _nd = {lbl:NormalizationData(lbl,dfsm[lbl],dfss[lbl],None,None) for lbl in _activecol}
            self._normdata.update(_nd)
            dfs = (dfs - dfsm)/dfss
            dfcat = dfout.filter(items=skip)
            dfout = pd.concat([dfs,dfcat],axis=1)
            return dfout
        dfsm = dfs.mean()
        dfss = dfs.std()
        _nd = {lbl:NormalizationData(lbl,dfsm[lbl],dfss[lbl],None,None) for lbl in dfin}
        self._normdata.update(_nd)
        return (dfout - dfsm)/dfss
    
    def _normalize_minmax(self,dfin,skip=[]):
        dfout = dfin.copy()
        if skip:
            _activecol = [col for col in dfin.columns if not col in skip]
            #print(_activecol)
            dfs = dfout.filter(items=_activecol)
            dfs = (dfs-dfs.min())/(dfs.max()-dfs.min())
            dfcat = dfout.filter(items=skip)
            dfout = pd.concat([dfs,dfcat],axis=1)
            return dfout
        return (dfout - dfout.min())/(dfout.max()-dfout.min())
        
    def _split_set(self,dfin,random_state=100,trng_frac=0.5):
        trng_feature = dfin[self.features].sample(random_state = random_state,frac=trng_frac)
        trng_target = dfin[self.target].sample(random_state = random_state,frac=trng_frac)
        test_feature = dfin[self.features].drop(trng_feature.index)
        test_target = dfin[self.target].drop(trng_target.index)
        return [trng_feature,trng_target,test_feature,test_target]
    
    def _prep_sets(self,ss_sets):
        
        ss_sets[0]=(np.concatenate((np.ones((ss_sets[0].shape[0],1)),ss_sets[0]),axis = 1))
        ss_sets[2]=(np.concatenate((np.ones((ss_sets[2].shape[0],1)),ss_sets[2]),axis = 1))
        #print(*ss_sets)    #print(s)
        return ss_sets

    def create_model(self,random_state:int,split:float,iterations=2500,alpha = 0.01,z_norm=[],min_max_norm=[]):
        #check target and features
        if (not self.target) or (not self.features) or (self.df.empty):
            raise Exception("Input sets not defined")
        self.df.dropna(inplace=True)
        #print("Available Data rows: ",self.df.shape[0])
        _typel = self.get_objt_cols(self.df)
        #print(_typel)
        _ocols = list(self.df.columns)
        _odf = self.df.copy()#omg this might take alot of memory :(
        
        if self.NormDefault == 'Z':                
            if min_max_norm:
                _zskip = [s for s in _ocols if s in min_max_norm or s in _typel]
                _mmskip = [s for s in _ocols if not s in min_max_norm or s in _typel]
                _odf = self._normalize_z(_odf,_zskip)
                _odf = self._normalize_minmax(_odf,_mmskip)
                self._p_w_Unnormalized([r for r in _ocols if r in _zskip or r in _mmskip and not r in _typel])
            elif z_norm:
                _zskip = [s for s in _ocols if not s in z_norm or s in _typel]
                _odf = self._normalize_z(_odf,_zskip)                
                self._p_w_Unnormalized([r for r in _ocols if r in _zskip and not r in _typel])
            else:
                _odf = self._normalize_z(_odf,_typel)
                
        elif self.NormDefault == 'MM':
            if z_norm:
                _mmskip = [s for s in _ocols if s in z_norm or s in _typel]
                _zskip = [s for s in _ocols if not s in z_norm or s in _typel]
                _odf = self._normalize_z(_odf,_zskip)
                _odf = self._normalize_minmax(_odf,_mmskip)
                self._p_w_Unnormalized([r for r in _ocols if r in _zskip or r in _mmskip and not r in _typel])
            elif min_max_norm:
                _mmskip = [s for s in _ocols if not s in min_max_norm or s in _typel]
                _odf = self._normalize_minmax(_odf,_mmskip)                
                self._p_w_Unnormalized([r for r in _ocols if r in _mmskip and not r in _typel])
            else:
                _odf = self._normalize_minmax(_odf,_typel)
        
        _ss = self._split_set(_odf,random_state,split)
        #print(*_ss)
        _ssc =[]
        for s in _ss:
            _ssc.append(self._categorize_strs_ohe(s,self.get_objt_cols(s)))
        
        feature_labels = list(_ssc[0].columns)
        _ssc = self._prep_sets(_ssc)
         
        #print(*_ssc)
        beta = np.zeros((_ssc[0].shape[1],1))
        #print("ba",beta)
        # Call the gradient_descent function
        beta, J_storage = self.gradient_descent_linreg(_ssc[0],_ssc[1].to_numpy(),beta,alpha,iterations)
        #get predictions against test set
        ypred = np.matmul(_ssc[2],beta)
        self._r2 = self._r2_score(_ssc[3],ypred)
        self._mse = self._mean_squared_error(_ssc[3],ypred)
        #print("R2: ",self._r2,"\nMSE: ",self._mse)
        res = MlrResults(self.features.copy(),feature_labels,self.target.copy(),beta.copy(),J_storage.copy(),_ssc[0].shape[0],self._r2.values[-1],self._mse.values[-1],iterations,alpha,split,random_state,self._normdata.copy())
        return res
        # call the predict() method
        #pred = predict_linreg(df_features_test,beta)
    
    def _r2_score(self,y, ypred):
        ybar = np.mean(y)
        return 1- (np.sum((y-ypred)**2,axis=0)/np.sum((y-ybar)**2,axis=0))

    def _mean_squared_error(self,y, pred):
        n = np.shape(y)[0]
        return np.sum((y-pred)**2,axis=0)/n

    def _calc_linreg(self,X,beta):
        #print("beta",beta)
        return np.matmul(X,beta)
    
    def compute_cost_mlinreg(self,X, y, beta):
        J = 0
        m = X.shape[0]
        _mat = (self._calc_linreg(X,beta) - y)
        J = 1/2/m*(np.matmul(np.transpose(_mat),_mat))
        return np.sum(J,axis=0)

    def gradient_descent_linreg(self,X, y, beta, alpha, num_iters):
        _xt=X.copy()
        _xt = np.transpose(_xt) #X T
        #print(_xt.shape)
        J_storage= np.zeros(num_iters)
        m = X.shape[0]
        for i in range(num_iters):
            #print(X)
            _y = self._calc_linreg(X,beta)
            #print("y", _y)
            beta = beta - np.matmul(alpha*(1/m)*_xt,_y-y)
            J_storage[i] = self.compute_cost_mlinreg(X,y,beta)
        return beta, J_storage

    def _err_f(self,y,yi):
        return (y-yi)**2
    
    def _p_w_Unnormalized(self,remaining):
        if remaining:
            print("Warning: Data has Un-normalized data",remaining)
        
    
        
            
        
        
        
            