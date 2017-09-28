import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr

train = pd.read_csv("./train.csv")

train_ID = train['Id']

#Drop 'Id' column since it's unnecessary for predictions
train.drop("Id", axis=1, inplace=True)

# Training set
ncols = train.shape[1] # Number of columns
nrows = train.shape[0] # Number of rows

#descriptive statistics summary
train['SalePrice'].describe()

# Kernel Density Plot
sns.distplot(train.SalePrice,fit=norm)
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
# QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
