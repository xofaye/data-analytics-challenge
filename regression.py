import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
from scipy.stats import skew, norm
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
import xgboost as xgb
import lightgbm as lgb

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

#log transform SalePrice
train["SalePrice"] = np.log1p(train["SalePrice"])
(mu, sigma) = norm.fit(train['SalePrice'])

# Drop outliers
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
train = train.drop(train['TotalBsmtSF'] > 6000)

# Combine test + train data for manipulation
all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                      test.loc[:, 'MSSubClass':'SaleCondition']))
all_data = all_data.drop(['Utilities'], axis=1)


# Find what data is missing
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

# Replace null values with None
for col in ('PoolQC', 'MiscFeature', 'GarageType', 'Alley', 'Fence', 'FireplaceQu', 'GarageFinish',
            'GarageQual', 'GarageCond', 'MasVnrType', 'MSSubClass', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
            'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

# Replace quantitative categories with 0
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'TotalBsmtSF',
            'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    all_data[col] = all_data[col].fillna(0)

# Replace categorical values by mode
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# These numberical values are actually categorical
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# One hot encoding
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

# Find skewed features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed_feats})
skewness = skewness[abs(skewness) > 0.75]
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

all_data = pd.get_dummies(all_data)

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
train = pd.DataFrame(all_data[:ntrain])
test = pd.DataFrame(all_data[ntrain:])

# From "Building Machine Learning Systems with Python" - Luis Pedro Coelho
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return(rmse)


all_data["PorchSF"] = all_data["OpenPorchSF"] + all_data["EnclosedPorch"] + all_data["3SsnPorch"] + all_data["ScreenPorch"]
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['BsmtFinSF'] = all_data['BsmtFinSF1'] + all_data['BsmtFinSF1']
#all_data = all_data.drop(['Utilities'], axis=1)
all_data = all_data.drop(['BsmtFinSF1'], axis=1)
all_data = all_data.drop(['BsmtFinSF2'], axis=1)
all_data = all_data.drop(['GarageCars'], axis=1)

train = pd.DataFrame(all_data[:ntrain])
test = pd.DataFrame(all_data[ntrain:])

# Lasso
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
# Ridge
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
# ElasticNet
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1, seed=7, nthread=-1)
# Light Gradient Boost Machine
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f}, STD: {:.4f}\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f}, STD: {:.4f}\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f}, STD: {:.4f}\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f}, STD: {:.4f}\n" .format(score.mean(), score.std()))


# Average base models
# from constt @ https://stats.stackexchange.com
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, x, y):
        self.models_ = [clone(z) for z in self.models]
        for model in self.models_:
            model.fit(x, y)
        return self

    def predict(self, x):
        predictions = np.column_stack([
            model.predict(x) for model in self.models_])
        #return np.average(predictions, axis=1, weights=[3./10, 2./10, 2./10, 3./10])
        # lol just tried different weights based on error score from above  ¯\_(ツ)_/¯
        #return np.average(predictions, axis=1, weights=[2./6, 1./6, 1./6, 2./6])
        return np.mean(predictions, axis=1)

# Averaged base models score
#averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
averaged_models = AveragingModels(models=(ENet, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f}, STD: {:.4f}\n".format(score.mean(), score.std()))


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

averaged_models.fit(train.values, y_train)
stacked_train_pred = averaged_models.predict(train.values)
stacked_pred = np.expm1(averaged_models.predict(test.values))
print("RMSLE ERROR FOR STACKED: ", rmsle(y_train, stacked_train_pred))

# XGBoost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

# LightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))


weights = [0.7, 0.15, 0.15]
print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*weights[0] +
               xgb_train_pred*weights[1] + lgb_train_pred*weights[2]))

# Ensembled Predictions:
ensemble = stacked_pred*weights[0] + xgb_pred*weights[1] + lgb_pred*weights[2]
#ensemble = stacked_pred*0.60 + xgb_pred*0.2 + lgb_pred*0.2

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv', index=False)
