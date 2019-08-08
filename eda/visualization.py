import seaborn as sns
import matplotlib.pyplot as plt
from index import *

def plot_sale_price_histogram():
    sns.distplot(train['SalePrice'])
    plt.show()

def plot_scatter_plots_for_vars(vars):
    for var in vars:
        data = pd.concat([train['SalePrice'], train[var]], axis=1)
        data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
        plt.show()

def boxplot_overall_quality():
    data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.show()

def boxplot_year_built():
    var = 'YearBuilt'
    data = pd.concat([train['SalePrice'], train[var]], axis=1)
    _, _ = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000);
    plt.xticks(rotation=90);
    plt.show()

# plot_sale_price_histogram()
# plot_scatter_plots_for_vars(['GrLivArea', 'TotalBsmtSF'])
# boxplot_overall_quality()
# boxplot_year_built()
