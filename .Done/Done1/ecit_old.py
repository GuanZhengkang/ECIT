from scipy import stats
from scipy.stats import norm
import numpy as np
from causallearn.utils.cit import CIT
from gammaEKCIT import gamma_KCIT


class ECIT():
    def __init__(self, data, alpha, typ, k=4, **kwargs):
        self.data = data
        self.k = k
        self.typ = typ
        self.alpha = alpha
        self.method = typ

    def __call__(self, X, Y, S=None, **kwargs):

        data = self.data
        k = self.k
        typ = self.typ
        alpha = self.alpha
        
        if typ == "ACAT": #ACAT
            shuffled_data = data[np.random.permutation(len(data))]
            pVal = []
            for sub_data in np.array_split(shuffled_data, k):
                kci_obj = CIT(sub_data, "kci", **kwargs)
                pVal.append(kci_obj(X, Y, S))
            t = np.mean(np.tan((np.abs(0.5-np.array(pVal)))*np.pi))
            return 0.5-np.arctan(t)/np.pi
    

        elif typ==None:
            kci_obj = CIT(data, "kci", **kwargs)
            return kci_obj(X, Y, S)
        
        elif typ=="Gamma":
            gamma_k = np.array(gamma_KCIT(data, X, Y, S, k=k))
            test_stat = np.sum(gamma_k[:,1])
            para_k = np.sum(gamma_k[:,2])
            theta = np.mean(gamma_k[:,3])
            pvalue = 1-stats.gamma.cdf(test_stat, para_k, 0, theta)
            return pvalue
                
        elif typ == "xACAT": #xACAT
            shuffled_data = data[np.random.permutation(len(data))]
            pVal = []
            for sub_data in np.array_split(shuffled_data, k):
                kci_obj = CIT(sub_data, "kci", **kwargs)
                pVal.append(kci_obj(X, Y, S))
            t = np.mean(5*np.tan((np.abs(0.5-np.array(pVal)))*np.pi))
            return 0.5-np.arctan(t)/np.pi
        
        elif typ=="bACAT": #Bootstrap
            pVal = []
            enpVal = []
            size = int(len(data)/k)
            for _ in range(k):
                boots_data = data[np.random.randint(0, len(data), size=size)]
                kci_obj = CIT(boots_data, "kci", **kwargs)
                pVal.append(kci_obj(X, Y, S))
                t = np.mean(np.tan((np.abs(0.5-np.array(pVal)))*np.pi))
                enpVal.append(0.5-np.arctan(t)/np.pi)
            while np.abs(np.mean(enpVal[:-1])-enpVal[-1])>0.01:
                boots_data = data[np.random.randint(0, len(data), size=size)]
                kci_obj = CIT(boots_data, "kci", **kwargs)
                pVal.append(kci_obj(X, Y, S))
                t = np.mean(np.tan((np.abs(0.5-np.array(pVal)))*np.pi))
                enpVal.append(0.5-np.arctan(t)/np.pi)
            return enpVal[-1]
        
        elif typ=="oACAT": #排序均分
            pVal = []
            sorted_data = data[data[:, 0].argsort()]
            for sub_data in [sorted_data[i::k] for i in range(k)]:
                kci_obj = CIT(sub_data, "kci", **kwargs)
                pVal.append(kci_obj(X, Y, S))
            t = np.mean(np.tan((np.abs(0.5-np.array(pVal)))*np.pi))
            return 0.5-np.arctan(t)/np.pi
        
        else:
            shuffled_data = data[np.random.permutation(len(data))]
            pVal = []
            for sub_data in np.array_split(shuffled_data, k):
                kci_obj = CIT(sub_data, "kci", **kwargs)
                pVal.append(kci_obj(X, Y, S))
            if typ == "a": #平均
                return np.mean(pVal)
            elif typ == "c": #投票
                if np.mean(np.array(pVal)<alpha)>0.5:
                    return 1e-15
                else:
                    return 0.5-(1e-15)
            elif typ == "c=":
                if np.mean(np.array(pVal)<alpha)>=0.5:
                    return 1e-15
                else:
                    return 0.5-(1e-15)
            elif typ == "cx":
                if np.mean(np.array(pVal)<alpha)>0.7:
                    return 1e-15
                else:
                    return 0.5-(1e-15)
            else: #测试
                t = np.mean(norm.ppf(1-np.array(pVal)))
                return 1-norm.cdf(t, loc=0, scale=1)

