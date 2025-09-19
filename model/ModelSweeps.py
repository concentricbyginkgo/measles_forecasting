###########################################################################
###   MODELSWEEPS.PY                                                    ###
###      * MASS MODEL IMPORTER FOR CROSS MODEL COMPARISONS              ###
###      * USES LAZY PREDICT MODELS WITH OTHER ADDITIONAL LIT MODELS    ###
###      * CAN TAKE GENERIC SKL CLASSES IF PASSED IN PYTHON             ###
###      * USES NEURAL PROPHET TO FORWARD PROJECT SEASONAL PREDICTORS   ###
###      * IMPLEMENTS GLOBAL-LOCAL MODELLING (MULTI GEOGRAPHY)          ###
###                                                                     ###
###             Contact: James Schlitt ~ jschlitt@ginkgobioworks.com    ###
###########################################################################


from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import BayesianRidge
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lars
from sklearn.linear_model import LarsCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from xgboost.sklearn import XGBRegressor


models = [{'model': BaggingRegressor, 'modelName': 'BaggingRegressor'},
          {'model': CatBoostRegressor, 'modelName': 'CatBoostRegressor'},
          {'model': DecisionTreeRegressor, 'modelName': 'DecisionTreeRegressor'},
          {'model': GaussianProcessRegressor, 'modelName': 'GaussianProcessRegressor'},
          {'model': GradientBoostingRegressor, 'modelName': 'GradientBoostingRegressor'},
          {'model': HistGradientBoostingRegressor, 'modelName': 'HistGradientBoostingRegressor'},
          {'model': HuberRegressor, 'modelName': 'HuberRegressor'},
          {'model': KNeighborsRegressor, 'modelName': 'KNeighborsRegressor'},
          {'model': Lars, 'modelName': 'Lars'},
          {'model': LarsCV, 'modelName': 'LarsCV'},
          {'model': Lasso, 'modelName': 'Lasso'},
          {'model': LassoCV, 'modelName': 'LassoCV'},
          {'model': LassoLars, 'modelName': 'LassoLars'},
          {'model': LassoLarsCV, 'modelName': 'LassoLarsCV'},
          {'model': LassoLarsIC, 'modelName': 'LassoLarsIC'},
          {'model': LinearSVR, 'modelName': 'LinearSVR'},
          {'model': NuSVR, 'modelName': 'NuSVR'},
          {'model': OrthogonalMatchingPursuit, 'modelName': 'OrthogonalMatchingPursuit'},
          {'model': OrthogonalMatchingPursuitCV, 'modelName': 'OrthogonalMatchingPursuitCV'},
          {'model': PassiveAggressiveRegressor, 'modelName': 'PassiveAggressiveRegressor'},
          {'model': RANSACRegressor, 'modelName': 'RANSACRegressor'},
          {'model': RandomForestRegressor, 'modelName': 'RandomForestRegressor'},
          {'model': Ridge, 'modelName': 'Ridge'},
          {'model': RidgeCV, 'modelName': 'RidgeCV'},
          {'model': TransformedTargetRegressor, 'modelName': 'TransformedTargetRegressor'},
          {'model': XGBRegressor, 'modelName': 'XGBRegressor'}]

removed = [{'model': AdaBoostRegressor, 'modelName': 'AdaBoostRegressor'},
          {'model': ExtraTreesRegressor, 'modelName': 'ExtraTreesRegressor'},
          {'model': SGDRegressor, 'modelName': 'SGDRegressor'},
          {'model': ElasticNet, 'modelName': 'ElasticNet'},
          {'model': ElasticNetCV, 'modelName': 'ElasticNetCV'},
          {'model': ExtraTreeRegressor, 'modelName': 'ExtraTreeRegressor'},
          {'model': DummyRegressor, 'modelName': 'DummyRegressor'},
          {'model': BayesianRidge, 'modelName': 'BayesianRidge'},
          {'model': KernelRidge, 'modelName': 'KernelRidge'},
          {'model': LinearRegression, 'modelName': 'LinearRegression'},
          {'model': SVR, 'modelName': 'SVR'},
          {'model': LGBMRegressor, 'modelName': 'LGBMRegressor'}]


regressorObjects = {str(model):model for model in models}
removedRegressorObjects = {str(model):model for model in removed}


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


classifiers = [{'model': AdaBoostClassifier, 'modelName': 'AdaBoostClassifier'},
               {'model': BaggingClassifier, 'modelName': 'BaggingClassifier'},
               {'model': BernoulliNB, 'modelName': 'BernoulliNB'},
               {'model': CalibratedClassifierCV, 'modelName': 'CalibratedClassifierCV'},
               {'model': DecisionTreeClassifier, 'modelName': 'DecisionTreeClassifier'},
               {'model': ExtraTreeClassifier, 'modelName': 'ExtraTreeClassifier'},
               {'model': ExtraTreesClassifier, 'modelName': 'ExtraTreesClassifier'},
               {'model': GaussianNB, 'modelName': 'GaussianNB'},
               {'model': GradientBoostingClassifier, 'modelName': 'GradientBoostingClassifier'},
               {'model': HistGradientBoostingClassifier, 'modelName': 'HistGradientBoostingClassifier'},
               {'model': KNeighborsClassifier, 'modelName': 'KNeighborsClassifier'},
               {'model': LabelPropagation, 'modelName': 'LabelPropagation'},
               {'model': LabelSpreading, 'modelName': 'LabelSpreading'},
               {'model': LinearSVC, 'modelName': 'LinearSVC'},
               {'model': LinearDiscriminantAnalysis, 'modelName': 'LinearDiscriminantAnalysis'},
               {'model': LogisticRegression, 'modelName': 'LogisticRegression'},
               {'model': LogisticRegressionCV, 'modelName': 'LogisticRegressionCV'},
               {'model': MLPClassifier, 'modelName': 'MLPClassifier'},
               {'model': NearestCentroid, 'modelName': 'NearestCentroid'},
               {'model': NuSVC, 'modelName': 'NuSVC'},
               {'model': PassiveAggressiveClassifier, 'modelName': 'PassiveAggressiveClassifier'},
               {'model': Perceptron, 'modelName': 'Perceptron'},
               {'model': QuadraticDiscriminantAnalysis, 'modelName': 'QuadraticDiscriminantAnalysis'},
               {'model': RidgeClassifier, 'modelName': 'RidgeClassifier'},
               {'model': RidgeClassifierCV, 'modelName': 'RidgeClassifierCV'},
               {'model': RandomForestClassifier, 'modelName': 'RandomForestClassifier'},
               {'model': SVC, 'modelName': 'SVC'},
               {'model': SGDClassifier, 'modelName': 'SGDClassifier'}]


classifierObjects = {str(classifier):classifier for classifier in classifiers}

