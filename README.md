# CRISP-DM-Methodology

## Business Understanding

- The client's primary objective is to better understand what factors make a car more or less valuable to consumers in order to competively price cars.
- To perform this task, the project will need access to the client's dataset, perform a data analysis after organizing and re-struturing the data if neccessary, and then create and assess different machine learning models following optimization to produce the most accurate model.
- The deliverable will be a supervised multiple regression machine learning model that produces a reasonable amount of error and is found to be most generalizable to future consumers with a report regarding how different features weigh on the algorithms.

## Data Understanding

- The dataset will be assessed for any missing, duplicative, or outlier values. Depending on the feature, the sample will be removed, replaced with the mean or median value, replaced with a fixed value, or no intervention will be performed if it is deemed that it will not affect the subsequent analysis.
- The quantitiative feature distributions will be assessed for any skewness and will be scaled.
- The categorical features will be assessed and may be further condensed if too many unneccesary values are found.

## Data Preparation 

- I now have a general impression of the data and developed strategies for feature engineering. 

- I have reviewed all of the feature names and I do not concur with their data types. I will switch year and odometer to integer. 

- I have 426880 observations and 18 features. 

- Of the 18 features, 14 are categorical and 4 are numeric.

- There are 1,215,152 missing entries/values.

- Due to their low proportions (<0.28-1.03% of the entire dataset), I will remove all samples that have missing values within the odometer and year features. If the proportions were higher than 5%, then I would have imputed the median instead of removing. The proportions are so low that it will not make a statistical difference either way.

- However, the features identified as condition, cylinders, VIN, drive, size, type, and paint_color are categorical values that cannot be reasonably imputed with mean or median values without grossly skewing their respective distributions. These missing values range from 21% to 71.8% of the entire dataset. I will remove the size feature because 71.8% of the dataset is missing this feature. I will further assess the remaining features and determine how I will engineer them but most likely will scale down the number of features down to a number that represent the bulk of the data within reason.

- I will remove the ID and VIN features from the dataset.

- There are no duplicate entries.

- I will transform the year feature so that a car from 2022 = 0 years and a car from 2000 has 22 years.

- I will use the 1.5 * min/max interquartile method to remove all outliers from year, price, and odomoter features. 

- If year and odomoter features have zero, they will remain in the dataset. However, if price has zero, then I wil remove these samples from the dataset.

- There are 42 different values in the manufacturer categorical feature.

- There are 404 different values in the region categorical feature.

- There are 29649 different values in the model categorical feature.

- I will further condense the manufacturer, region, and model categorical features.

- The descriptive statistics do not appear to be reasonable at the outset due to outliers. 

- The range in prices is from 0(?) to 373,6929,000(?) dollars. This feature will be reviewed and all outliers will be removed.

- The range in year starts in 1900? Such outliers will also be removed from the dataset and will be converted to how old is the car.

- The maximum odomoter value is 10 million miles? This feature will be reviewed and outliers will be removed.

## Evaluation

### Model Selection

After performing and evaluating 4 different models, below is their ranking method:

1. Model 4|RandomSearchCV, Target Encoding, & Random Forest Regression with Standard Scaling:
R2: 89.49%
Mean Absolute Error: $2229.48
Root Mean Square Training Error: $2153.66
Root Mean Square Testing Error: $4124.540
2. Model 1|CV Grid Search, Target Encoding, Ridge Regression, & Standard Scaling:
R2: 62.39%
Mean Absolute Error: $5638.07
Root Mean Square Training Error: $7876.89
Root Mean Square Testing Error: $7801.09
3. Model 3|GridSearchCV, Target Encoding, Sequential Selection with Lasso, Ridge Regression, & Standard Scaling
R2: 57.18%
Mean Absolute Error: $6071.51
Root Mean Square Training Error: $8410.42
Root Mean Square Testing Error: $8324.4
4. Model2| CV Grid Search, Target Encoding, Logarithmic Ridge Regression, & Standard Scaling
R2: 52.96%
Mean Absolute Error: $6066.64
Root Mean Square Training Error: $8785.36
Root Mean Square Testing Error: $8724.93

### Feature Importance
The model's feature permutation importances are reported for models 1-3 and the feature importance for model4:

1. Model 4|RandomSearchCV, Target Encoding, & Random Forest Regression with Standard Scaling:
year	0.456100
type	0.115429
odometer	0.102375
fuel	0.078583
cylinders	0.065237
drive	0.052988
model	0.040866
manufacturer	0.025039
state	0.016531
paint_color	0.011242
condition	0.011218
title_status	0.009286
region	0.008315
transmission	0.006790

2. Model 1|CV Grid Search, Target Encoding, Ridge Regression, & Standard Scaling:
year   2565.560 +/- 12.345
odometer   1198.709 +/- 9.292
drive   409.022 +/- 5.865
fuel   376.377 +/- 6.247
type   368.684 +/- 5.561
cylinders   244.056 +/- 4.572
model   238.369 +/- 5.004
title_status   93.191 +/- 2.930
state   62.022 +/- 2.362
manufacturer   40.381 +/- 2.402
region   18.656 +/- 1.307
paint_color   13.745 +/- 1.257
condition   8.213 +/- 1.102

3. Model 3|GridSearchCV, Target Encoding, Sequential Selection with Lasso, Ridge Regression, & Standard Scaling
year   2564.218 +/- 12.671
odometer   1128.481 +/- 8.155
type   1007.692 +/- 8.594
cylinders   736.721 +/- 6.448
fuel   479.011 +/- 6.563

4. Model2| CV Grid Search, Target Encoding, Logarithmic Ridge Regression, & Standard Scaling
year   2431.489 +/- 10.389
odometer   746.893 +/- 4.718
type   490.505 +/- 7.577
drive   242.928 +/- 3.426
fuel   239.821 +/- 6.496
model   228.746 +/- 4.686
cylinders   214.804 +/- 5.057
title_status   80.465 +/- 2.997
state   67.744 +/- 3.601
paint_color   63.709 +/- 3.442
condition   49.236 +/- 3.490
region   38.880 +/- 2.720
manufacturer   20.802 +/- 1.641
transmission   7.855 +/- 0.376

- All models concur that the most important feature is year
- Odometer is in the second most important feature in 3/4 models
- Type is in the top 3 features in 3/4 models
- Fuel is in the top 5 features in all 4 models
- Cylinders and perhpas model are the only remaining feature that carrys any significant remaining effect

### Data Integrity

- The final dataset titled prepped_cars represents 85.29% of the original dataset yet only has 48.80% of the null values compared to the original dataset. It is the projects team's professional opinion that removing approximately 15% of the original dataset to decrease the amout of missing data by 51.2% is a reasonable trade-off. 
 
- It is also the project team's professional opinion that fundamental changes to the underlying distributions of the following categorical features would occur if the project simply replaced the missing values with the most frequent categorical value: conidition, cylinders, drive, type, and paint_color.

- These fundamental changes could result in a scenario where the learning algoirithms would demonstrate improved accuracy and decreased training and testing error. 

- However, this "improved performance" on a dataset that has been grossly manipulated would be poorly generalizable from a deployment perspective and could result in a model that is much less robust to actual real-world practices.

- A firm recommendation from the project team is for the dealership to devote resources to improve data collection with the goal of improving future data projects.

- However, since the current data project was able to create a model with a R2 of approximately 90%, such an intervention may not be neccesary as long as current data collection practices produce similar datasets in the future.

- Indeed, a Random Forest algorithm is known to handle datasets that are both non-linear and categorical well. This supervised machine learning project demonstrated the limits of multiple regression and the strengths of ensemble methodolgy due to the numerous categorical features and missing values of the dataset.

## Deployment
### Deployment Plan

- The model will be deployed as a web application on a local dealership machine so that the client's dealership managers or assigned staff can competitively price their used car inventory or make appropriate offers when offering trade-in value.
- The model's pricing recommendation will be contrasted with the dealership's usual process of pricing inventory.
- Depending on how investigative the dealership would like to become concerning the model's robustness, the dealership could opt for control and treatment groups. Where the control group are dealerships that continue to use current pricing practices and the treatment group are dealerships that adheres only to the model. 
- A between groups analysis can be performed after a pre-determined period of time and appropriate metrics such as profit, volume, lot time, repeat customer, etc., could be evaluated to measure any model effect between the control and treatment groups.
- The general manager can notify their regional manager of any gross inaccuracies between the model and the actual sale price, overrule the model's recommendation, and a detailed report will be collected and forwarded to the chief information/analytics officer for further review, monitoring, and data mining engagement.
