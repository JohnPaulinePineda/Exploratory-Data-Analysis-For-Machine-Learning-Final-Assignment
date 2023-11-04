***
# Data Preprocessing : Data Quality Assessment, Preprocessing and Exploration for a Regression Modelling Problem

***
### John Pauline Pineda <br> <br> *November 7, 2023*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
        * [1.4.1 Missing Data Imputation](#1.4.1)
        * [1.4.2 Outlier Treatment](#1.4.2)
        * [1.4.3 Zero and Near-Zero Variance](#1.4.3)
        * [1.4.4 Collinearity](#1.4.4)
        * [1.4.5 Linear Dependencies](#1.4.5)
        * [1.4.6 Centering and Scaling](#1.4.6)
        * [1.4.7 Shape Transformation](#1.4.7)
        * [1.4.8. Dummy Variables](#1.4.8)
        * [1.4.9. Preprocessed Data Description](#1.4.9)
     * [1.5 Data Exploration](#1.5)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project explores the various methods in assessing **Data Quality**, implementing **Data Preprocessing** and conducting **Data Exploration** for prediction problems with numeric responses using various helpful packages in <mark style="background-color: #CCECFF">**Python**</mark>. A non-exhaustive list of methods to detect missing data, extreme outlying points, near-zero variance, multicollinearity, linear dependencies and skewed distributions were evaluated. Remedial procedures on addressing data quality issues including missing data imputation, centering and scaling transformation, shape transformation and outlier treatment were similarly considered, as applicable. All results were consolidated in a [<span style="color: #FF0000">**Summary**</span>](#Summary) presented at the end of the document.

[Data quality assessment](http://appliedpredictivemodeling.com/) involves profiling and assessing the data to understand its suitability for machine learning tasks. The quality of training data has a huge impact on the efficiency, accuracy and complexity of machine learning tasks. Data remains susceptible to errors or irregularities that may be introduced during collection, aggregation or annotation stage. Issues such as incorrect labels, synonymous categories in a categorical variable or heterogeneity in columns, among others, which might go undetected by standard pre-processing modules in these frameworks can lead to sub-optimal model performance, inaccurate analysis and unreliable decisions.

[Data preprocessing](http://appliedpredictivemodeling.com/) involves changing the raw feature vectors into a representation that is more suitable for the downstream modelling and estimation processes, including data cleaning, integration, reduction and transformation. Data cleaning aims to identify and correct errors in the dataset that may negatively impact a predictive model such as removing outliers, replacing missing values, smoothing noisy data, and correcting inconsistent data. Data integration addresses potential issues with redundant and inconsistent data obtained from multiple sources through approaches such as detection of tuple duplication and data conflict. The purpose of data reduction is to have a condensed representation of the data set that is smaller in volume, while maintaining the integrity of the original data set. Data transformation converts the data into the most appropriate form for data modeling.

[Data exploration](http://appliedpredictivemodeling.com/) involves analyzing and investigating data sets to summarize their main characteristics, often employing data visualization methods. It helps determine how best to manipulate data sources to discover patterns, spot anomalies, test a hypothesis, or check assumptions. This process is primarily used to see what data can reveal beyond the formal modeling or hypothesis testing task and provides a better understanding of data set variables and the relationships between them.

## 1.1. Data Background <a class="anchor" id="1.1"></a>

Dataset used for the analysis was separately gathered and consolidated from various sources including: 
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


## 1.2. Data Description <a class="anchor" id="1.2"></a>

The dataset is comprised of:
* **177 rows** (observations)
* **22 columns** (variables)
    * **1/22 metadata** (categorical)
        * **COUNTRY**
    * **1/22 target** (numeric)
         * **CANRAT**
    * **19/22 predictor** (numeric)
         * **GDPPER**
         * **URBPOP**
         * **PATRES**
         * **RNDGDP**
         * **POPGRO**
         * **LIFEXP**
         * **TUBINC**
         * **DTHCMD**
         * **AGRLND**
         * **GHGEMI**
         * **RELOUT**
         * **METEMI**
         * **FORARE**
         * **CO2EMI**
         * **PM2EXP**
         * **POPDEN**
         * **GDPCAP**
         * **ENRTER**
         * **EPISCO**
     * **1/22 predictor** (categorical)
         * **HDICAT**


```python
##################################
# Loading Python Libraries
##################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
##################################
# Loading dataset
##################################
cancer_rate = pd.read_csv('CancerRates.csv')
```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ', cancer_rate.shape)
```

    Dataset Dimensions:  (177, 22)
    


```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:\n',cancer_rate.dtypes)
```

    Column Names and Data Types:
     COUNTRY     object
    CANRAT     float64
    GDPPER     float64
    URBPOP     float64
    PATRES     float64
    RNDGDP     float64
    POPGRO     float64
    LIFEXP     float64
    TUBINC     float64
    DTHCMD     float64
    AGRLND     float64
    GHGEMI     float64
    RELOUT     float64
    METEMI     float64
    FORARE     float64
    CO2EMI     float64
    PM2EXP     float64
    POPDEN     float64
    ENRTER     float64
    GDPCAP     float64
    HDICAT      object
    EPISCO     float64
    dtype: object
    


```python
##################################
# Taking a snapshot of the dataset
##################################
cancer_rate.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>COUNTRY</th>
      <th>CANRAT</th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>PATRES</th>
      <th>RNDGDP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>...</th>
      <th>RELOUT</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>ENRTER</th>
      <th>GDPCAP</th>
      <th>HDICAT</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>452.4</td>
      <td>98380.63601</td>
      <td>86.241</td>
      <td>2368.0</td>
      <td>NaN</td>
      <td>1.235701</td>
      <td>83.200000</td>
      <td>7.2</td>
      <td>4.941054</td>
      <td>...</td>
      <td>13.637841</td>
      <td>131484.763200</td>
      <td>17.421315</td>
      <td>14.772658</td>
      <td>24.893584</td>
      <td>3.335312</td>
      <td>110.139221</td>
      <td>51722.06900</td>
      <td>VH</td>
      <td>60.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New Zealand</td>
      <td>422.9</td>
      <td>77541.76438</td>
      <td>86.699</td>
      <td>348.0</td>
      <td>NaN</td>
      <td>2.204789</td>
      <td>82.256098</td>
      <td>7.2</td>
      <td>4.354730</td>
      <td>...</td>
      <td>80.081439</td>
      <td>32241.937000</td>
      <td>37.570126</td>
      <td>6.160799</td>
      <td>NaN</td>
      <td>19.331586</td>
      <td>75.734833</td>
      <td>41760.59478</td>
      <td>VH</td>
      <td>56.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>372.8</td>
      <td>198405.87500</td>
      <td>63.653</td>
      <td>75.0</td>
      <td>1.23244</td>
      <td>1.029111</td>
      <td>82.556098</td>
      <td>5.3</td>
      <td>5.684596</td>
      <td>...</td>
      <td>27.965408</td>
      <td>15252.824630</td>
      <td>11.351720</td>
      <td>6.768228</td>
      <td>0.274092</td>
      <td>72.367281</td>
      <td>74.680313</td>
      <td>85420.19086</td>
      <td>VH</td>
      <td>57.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>362.2</td>
      <td>130941.63690</td>
      <td>82.664</td>
      <td>269586.0</td>
      <td>3.42287</td>
      <td>0.964348</td>
      <td>76.980488</td>
      <td>2.3</td>
      <td>5.302060</td>
      <td>...</td>
      <td>13.228593</td>
      <td>748241.402900</td>
      <td>33.866926</td>
      <td>13.032828</td>
      <td>3.343170</td>
      <td>36.240985</td>
      <td>87.567657</td>
      <td>63528.63430</td>
      <td>VH</td>
      <td>51.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Denmark</td>
      <td>351.1</td>
      <td>113300.60110</td>
      <td>88.116</td>
      <td>1261.0</td>
      <td>2.96873</td>
      <td>0.291641</td>
      <td>81.602439</td>
      <td>4.1</td>
      <td>6.826140</td>
      <td>...</td>
      <td>65.505925</td>
      <td>7778.773921</td>
      <td>15.711000</td>
      <td>4.691237</td>
      <td>56.914456</td>
      <td>145.785100</td>
      <td>82.664330</td>
      <td>60915.42440</td>
      <td>VH</td>
      <td>77.9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
##################################
# Performing a general exploration of the numeric variables
##################################
print('Numeric Variable Summary: \n', cancer_rate.describe(include='number').transpose())
```

    Numeric Variable Summary: 
             count           mean           std          min           25%  \
    CANRAT  177.0     183.829379  7.974340e+01    78.400000    118.100000   
    GDPPER  165.0   45284.424283  3.941794e+04  1718.804896  13545.254510   
    URBPOP  174.0      59.788121  2.280640e+01    13.345000     42.432750   
    PATRES  108.0   20607.388889  1.340683e+05     1.000000     35.250000   
    RNDGDP   74.0       1.197474  1.189956e+00     0.039770      0.256372   
    POPGRO  174.0       1.127028  1.197718e+00    -2.079337      0.236900   
    LIFEXP  174.0      71.746113  7.606209e+00    52.777000     65.907500   
    TUBINC  174.0     105.005862  1.367229e+02     0.770000     12.000000   
    DTHCMD  170.0      21.260521  1.927333e+01     1.283611      6.078009   
    AGRLND  174.0      38.793456  2.171551e+01     0.512821     20.130276   
    GHGEMI  170.0  259582.709895  1.118550e+06   179.725150  12527.487367   
    RELOUT  153.0      39.760036  3.191492e+01     0.000296     10.582691   
    METEMI  170.0   47876.133575  1.346611e+05    11.596147   3662.884908   
    FORARE  173.0      32.218177  2.312001e+01     0.008078     11.604388   
    CO2EMI  170.0       3.751097  4.606479e+00     0.032585      0.631924   
    PM2EXP  167.0      91.940595  2.206003e+01     0.274092     99.627134   
    POPDEN  174.0     200.886765  6.453834e+02     2.115134     27.454539   
    ENRTER  116.0      49.994997  2.970619e+01     2.432581     22.107195   
    GDPCAP  170.0   13992.095610  1.957954e+04   216.827417   1870.503029   
    EPISCO  165.0      42.946667  1.249086e+01    18.900000     33.000000   
    
                     50%            75%           max  
    CANRAT    155.300000     240.400000  4.524000e+02  
    GDPPER  34024.900890   66778.416050  2.346469e+05  
    URBPOP     61.701500      79.186500  1.000000e+02  
    PATRES    244.500000    1297.750000  1.344817e+06  
    RNDGDP      0.873660       1.608842  5.354510e+00  
    POPGRO      1.179959       2.031154  3.727101e+00  
    LIFEXP     72.464610      77.523500  8.456000e+01  
    TUBINC     44.500000     147.750000  5.920000e+02  
    DTHCMD     12.456279      36.980457  6.520789e+01  
    AGRLND     40.386649      54.013754  8.084112e+01  
    GHGEMI  41009.275980  116482.578575  1.294287e+07  
    RELOUT     32.381668      63.011450  1.000000e+02  
    METEMI  11118.976025   32368.909040  1.186285e+06  
    FORARE     31.509048      49.071780  9.741212e+01  
    CO2EMI      2.298368       4.823496  3.172684e+01  
    PM2EXP    100.000000     100.000000  1.000000e+02  
    POPDEN     77.983133     153.993650  7.918951e+03  
    ENRTER     53.392460      71.057467  1.433107e+02  
    GDPCAP   5348.192875   17421.116227  1.173705e+05  
    EPISCO     40.900000      50.500000  7.790000e+01  
    


```python
##################################
# Performing a general exploration of the categorical variable
##################################
print('Categorical Variable Summary: \n', cancer_rate.describe(include='object').transpose())
```

    Categorical Variable Summary: 
             count unique        top freq
    COUNTRY   177    177  Australia    1
    HDICAT    167      4         VH   59
    

## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>
Details


```python

```


```python

```


```python


```


```python

```


```python

```


```python

```

## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>
Details

### 1.4.1 Missing Data Imputation <a class="anchor" id="1.4.1"></a>
Details

### 1.4.2 Outlier Treatment <a class="anchor" id="1.4.2"></a>
Details

### 1.4.3 Zero and Near-Zero Variance <a class="anchor" id="1.4.3"></a>
This is sub section 1.3.3

### 1.4.4 Collinearity <a class="anchor" id="1.4.4"></a>
Details

### 1.4.5 Linear Dependencies <a class="anchor" id="1.4.5"></a>
Details

### 1.4.6 Centering and Scaling <a class="anchor" id="1.4.6"></a>
Details

### 1.4.7 Shape Transformation <a class="anchor" id="1.4.7"></a>
Details

### 1.4.8 Dummy Variables <a class="anchor" id="1.4.8"></a>
Details

### 1.4.9 Preprocessed Data Description <a class="anchor" id="1.4.9"></a>
Details

## 1.5. Data Exploration <a class="anchor" id="1.5"></a>
Details

# 2. Summary <a class="anchor" id="Summary"></a>
Details

# 3. References <a class="anchor" id="References"></a>
* **[Book]** [Data Preparation for Machine Learning: Data Cleaning, Feature Selection, and Data Transforms in Python](https://machinelearningmastery.com/data-preparation-for-machine-learning/) by Jason Brownlee
* **[Book]** [Feature Engineering and Selection: A Practical Approach for Predictive Models](http://www.feat.engineering/) by Max Kuhn and Kjell Johnson
* **[Book]** [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) by Alice Zheng and Amanda Casari
* **[Book]** [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) by Max Kuhn and Kjell Johnson
* **[Book]** [Data Mining: Practical Machine Learning Tools and Techniques](https://www.sciencedirect.com/book/9780123748560/data-mining-practical-machine-learning-tools-and-techniques?via=ihub=) by Ian Witten, Eibe Frank, Mark Hall and Christopher Pal 
* **[Book]** [Data Cleaning](https://dl.acm.org/doi/book/10.1145/3310205) by Ihab Ilyas and Xu Chu
* **[Book]** [Data Wrangling with Python](https://www.oreilly.com/library/view/data-wrangling-with/9781491948804/) by Jacqueline Kazil and Katharine Jarmul
* **[Python Library API]** [sklearn.datasets.make classification API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.preprocessing.MinMaxScaler API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.model selection.train test split API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.linear model.LogisticRegression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.model selection.RepeatedStratifiedKFold API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.model selection.cross val score API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) by Scikit-Learn Team

***


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

