# EX-06 FEATURE TRANSFORMATION
## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.
## Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## Algorithm:
Step1: Read the given Data.
<br>
Step2: Clean the Data Set using Data Cleaning Process.
<br>
Step3: Apply Feature Transformation techniques to all the features of the data set.
<br>
Step4: Print the transformed features.
<br>
## Program:
```
Developed By: Ragavendran A
Register No: 212222230114
```
```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
df.skew()

np.log(df["Highly Positive Skew"])
np.reciprocal(df["Moderate Positive Skew"])
np.sqrt(df["Highly Positive Skew"])
np.square(df["Highly Positive Skew"])
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
df["Moderate Positive Skew_boxcox"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

import matplotlib.pyplot as plt
import statsmodels.api as sm
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
sm.qqplot(df['Moderate Negative Skew_1'],line='45')
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
## OUTPUT

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/c87c186e-aa56-4ada-bd0b-f206e28da12c)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/79e6bbba-c80b-4e4f-936e-90548808b4fe)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/981ffa46-f0e2-4064-8300-cabc49f1fe29)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/e13a50a8-3720-4fbb-ad5a-f53e8f073f62)

![image](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/82d25927-e0a0-40da-a568-29fcf65fb1cd)

![1](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/974244ba-bf85-49a2-9c5a-9efc6917f089)

![2](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/6f5c1e47-de3d-488c-a0a9-e6a3e318cec5)

![3](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/221b51d9-69c6-417c-8eda-586a225fa011)

![4](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/7555347c-e9d6-45b9-ba5e-ecfc517a6d6a)

![5](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/218fa41e-8ecd-41a0-b2e5-077e02563ab5)

![6](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/ea744e15-ca1e-44a4-bad6-97f3cc44ed55)

![7](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/85935965-3f8d-4272-a6f8-246d5cf51446)

![8](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/9140db78-3b28-4aa2-b4cf-7aa331cc4089)

![9](https://github.com/kavinesh8476/ODD2023-Datascience-Ex06/assets/118466561/36bc734c-ca61-4e86-bd30-20915f1fd7b4)



## Result:
Thus feature transformation is done for the given dataset.
