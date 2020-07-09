import numpy as np
class SimpleLinearRegressino.1:
    def __init__(self):
        self.a_ = None
        self.b_ = None
    def fit(self,x_train,y_train):
        assert x_train.ndim == 1,\
            "Simple linear regressor can only solve single feature training data."
        assert len(x_train) == len(y_train),\
            "the size x_train must be equal to the size of y_train."
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num = 0.0
        d = 0.0
        for i,j in x_train,y_train:
            num += (i-x_mean)*(j-y_mean)
            d += (i-x_mean)**2
        self.a_ = num + d
        self.b_ = y_mean-self.a_*x_mean
        return self
    def predict(self,x_predict):
        assert x_predict.ndim == 1 ,\
            "Simple linear regressor can only solve single feature training data."
        assert  self.a_ is not None and self.b_ is not None,\
            "must fit before predice"
        return [self._predict(x) for x in x_predict]
    def _predict(self,x_single):
        return [self.a_ * x_single + self.b_]
    def __repr__(self):
        return "SimpleLinearRegressino.1"