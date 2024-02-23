from MultiLinearReg import *
import numpy as np
class PredictorMLR():
    def __init__(self,model_f,normdata_label):
        self.model = pd.read_csv(model_f)
        self.targets={}
        for row in self.model.iterrows():
            print(row)
            self.targets[row['cat']]=row.copy().drop('cat')
        print(self.targets)
        
    