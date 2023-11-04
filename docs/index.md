***
# Data Preprocessing : Data Quality Assessment, Preprocessing and Exploration for a Regression Modelling Problem

***
### John Pauline Pineda <br> <br> *November 7, 2023*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Sample Data](#1.1)
    * [1.2 Data Quality Assessment](#1.2)
    * [1.3 Data Preprocessing](#1.3)
        * [1.3.1 Missing Data Imputation](#1.3.1)
        * [1.3.2 Outlier Treatment](#1.3.2)
        * [1.3.3 Zero and Near-Zero Variance](#1.3.3)
        * [1.3.4 Collinearity](#1.3.4)
        * [1.3.5 Linear Dependencies](#1.3.5)
        * [1.3.6 Centering and Scaling](#1.3.6)
        * [1.3.7 Shape Transformation](#1.3.7)
        * [1.3.8. Dummy Variables](#1.3.8)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project explores the various methods in assessing **Data Quality**, implementing **Data Preprocessing** and conducting **Data Exploration** for prediction problems with numeric responses using various helpful packages in <mark style="background-color: #CCECFF">**Python**</mark>. A non-exhaustive list of methods to detect missing data, extreme outlying points, near-zero variance, multicollinearity, linear dependencies and skewed distributions were evaluated. Remedial procedures on addressing data quality issues including missing data imputation, centering and scaling transformation, shape transformation and outlier treatment were similarly considered, as applicable. All results were consolidated in a [<span style="color: #FF0000">**Summary**</span>](#Summary) presented at the end of the document.

[Data quality assessment](http://appliedpredictivemodeling.com/) involves profiling and assessing the data to understand its suitability for machine learning tasks. The quality of training data has a huge impact on the efficiency, accuracy and complexity of machine learning tasks. Data remains susceptible to errors or irregularities that may be introduced during collection, aggregation or annotation stage. Issues such as incorrect labels, synonymous categories in a categorical variable or heterogeneity in columns, among others, which might go undetected by standard pre-processing modules in these frameworks can lead to sub-optimal model performance, inaccurate analysis and unreliable decisions.

[Data preprocessing](http://appliedpredictivemodeling.com/) involves changing the raw feature vectors into a representation that is more suitable for the downstream modelling and estimation processes, including data cleaning, integration, reduction and transformation. Data cleaning aims to identify and correct errors in the dataset that may negatively impact a predictive model such as removing outliers, replacing missing values, smoothing noisy data, and correcting inconsistent data. Data integration addresses potential issues with redundant and inconsistent data obtained from multiple sources through approaches such as detection of tuple duplication and data conflict. The purpose of data reduction is to have a condensed representation of the data set that is smaller in volume, while maintaining the integrity of the original data set. Data transformation converts the data into the most appropriate form for data modeling.

[Data exploration](http://appliedpredictivemodeling.com/) involves analyzing and investigating data sets to summarize their main characteristics, often employing data visualization methods. It helps determine how best to manipulate data sources to discover patterns, spot anomalies, test a hypothesis, or check assumptions. This process is primarily used to see what data can reveal beyond the formal modeling or hypothesis testing task and provides a better understanding of data set variables and the relationships between them.

## 1.1. Sample Data <a class="anchor" id="1.1"></a>

Dataset used for the analysis was manually gathered and consolidated from various sources including: 
1. Cancer Rates from [World Population Review](https://worldpopulationreview.com/country-rankings/cancer-rates-by-country)
2. Social Protection and Labor Indicator from [World Bank](https://data.worldbank.org/topic/social-protection-and-labor?view=chart)
3. Education Indicator from [World Bank](https://data.worldbank.org/topic/education?view=chart)
4. Economy and Growth Indicator from [World Bank](https://data.worldbank.org/topic/economy-and-growth?view=chart)
5. Environment Indicator from [World Bank](https://data.worldbank.org/topic/environment?view=chart)
6. Climate Change Indicator from [World Bank](https://data.worldbank.org/topic/climate-change?view=chart)
7. Agricultural and Rural Development Indicator from [World Bank](https://data.worldbank.org/topic/agriculture-and-rural-development?view=chart)
8. Social Development Indicator from [World Bank](https://data.worldbank.org/topic/social-development?view=chart)
9. Health Indicator from [World Bank](https://data.worldbank.org/topic/health?view=chart)
10. Science and Technology Indicator from [World Bank](https://data.worldbank.org/topic/science-and-technology?view=chart)
11. Urban Development Indicator from [World Bank](https://data.worldbank.org/topic/urban-development?view=chart)
12. Social Protection and Labor Indicator from [World Bank](https://data.worldbank.org/topic/social-protection-and-labor?view=chart)
13. Human Development Indices from [Human Development Reports](https://hdr.undp.org/data-center/human-development-index#/indicies/HDI)
14. Environmental Performance Indices from [Yale Center for Environmental Law and Policy](https://epi.yale.edu/epi-results/2022/component/epi)

This study hypothesized that various global development indicators and indices influence cancer rates across countries.

The target variable for the study is:
* **CANRAT** - Age-standardized cancer rates, per 100K population (2022)

The predictor variables for the study are:
* **GDPPER** - GDP per person employed, current US Dollars (2020)
* **URBPOP** - Urban population, % of total population (2020)
* **PATRES** - Patent applications by residents, total count (2020)
* **RNDGDP** - Research and development expenditure, % of GDP (2020)
* **POPGRO** - Population growth, annual % (2020)
* **LIFEXP** - Life expectancy at birth, total in years (2020)
* **TUBINC** - Incidence of tuberculosis, per 100K population (2020)
* **DTHCMD** - Cause of death by communicable diseases and maternal, prenatal and nutrition conditions,  % of total (2019)
* **AGRLND** - Agricultural land,  % of land area (2020)
* **GHGEMI** - Total greenhouse gas emissions, kt of CO2 equivalent (2020)
* **RELOUT** - Renewable electricity output, % of total electricity output (2015)
* **METEMI** - Methane emissions, kt of CO2 equivalent (2020)
* **FORARE** - Forest area, % of land area (2020)
* **CO2EMI** - CO2 emissions, metric tons per capita (2020)
* **PM2EXP** - PM2.5 air pollution, population exposed to levels exceeding WHO guideline value,  % of total (2017)
* **POPDEN** - Population density, people per sq. km of land area (2020)
* **GDPCAP** - GDP per capita, current US Dollars (2020)
* **ENRTER** - Tertiary school enrollment, % gross (2020)
* **HDICAT** - Human development index, ordered category (2020)
* **EPISCO** - Environment performance index , score (2022)


## 1.2. Data Quality Assessment <a class="anchor" id="1.2"></a>
This is section 1.2


```python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

## 1.3. Data Preprocessing <a class="anchor" id="1.3"></a>
This is section 1.3

### 1.3.1 Missing Data Imputation <a class="anchor" id="1.3.1"></a>
This is sub section 1.3.1

### 1.3.2 Outlier Treatment <a class="anchor" id="1.3.2"></a>
This is sub section 1.3.2

### 1.3.3 Zero and Near-Zero Variance <a class="anchor" id="1.3.3"></a>
This is sub section 1.3.3

### 1.3.4 Collinearity <a class="anchor" id="1.3.4"></a>
This is sub section 1.3.4

### 1.3.5 Linear Dependencies <a class="anchor" id="1.3.5"></a>
This is sub section 1.3.5

### 1.3.6 Centering and Scaling <a class="anchor" id="1.3.6"></a>
This is sub section 1.3.6

### 1.3.7 Shape Transformation <a class="anchor" id="1.3.7"></a>
This is sub section 1.3.7

### 1.3.8 Dummy Variables <a class="anchor" id="1.3.8"></a>
This is sub section 1.3.8

# 2. Summary <a class="anchor" id="Summary"></a>
This is Summary

# 3. References <a class="anchor" id="References"></a>
* **[Book]** [Data Preparation for Machine Learning: Data Cleaning, Feature Selection, and Data Transforms in Python](https://machinelearningmastery.com/data-preparation-for-machine-learning/) by Jason Brownlee
* **[Book]** [Feature Engineering and Selection: A Practical Approach for Predictive Models](http://www.feat.engineering/) by Max Kuhn and Kjell Johnson
* **[Book]** [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) by Alice Zheng and Amanda Casari
* **[Book]** [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) by Max Kuhn and Kjell Johnson
* **[Book]** [Data Mining: Practical Machine Learning Tools and Techniques](https://www.sciencedirect.com/book/9780123748560/data-mining-practical-machine-learning-tools-and-techniques?via=ihub=) by Ian Witten, Eibe Frank, Mark Hall and Christopher Pal 
* **[Book]** [Data Cleaning](https://dl.acm.org/doi/book/10.1145/3310205) by Ihab Ilyas and Xu Chu
* **[Book]** [Data Wrangling with Python](https://www.oreilly.com/library/view/data-wrangling-with/9781491948804/) by Jacqueline Kazil and Katharine Jarmul
* **[Python Library|API]** [sklearn.datasets.make classification API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) by Scikit-Learn Team
* **[Python Library|API]** [sklearn.preprocessing.MinMaxScaler API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) by Scikit-Learn Team
* **[Python Library|API]** [sklearn.model selection.train test split API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) by Scikit-Learn Team
* **[Python Library|API]** [sklearn.linear model.LogisticRegression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) by Scikit-Learn Team
* **[Python Library|API]** [sklearn.model selection.RepeatedStratifiedKFold API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html) by Scikit-Learn Team
* **[Python Library|API]** [sklearn.model selection.cross val score API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) by Scikit-Learn Team

***


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

