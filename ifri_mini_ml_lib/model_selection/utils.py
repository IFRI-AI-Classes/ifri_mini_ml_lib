

class BaseEstimatorMinimal:
    """A Base Estimator which implements minimalist methods get_params() and set_params()"""
    def get_params(self, deep=True):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self