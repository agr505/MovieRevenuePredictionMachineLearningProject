# Project title: Movie Revenue Prediction
## Team members: Sanmesh, Aaron, Tarushree, Aastha, Prithvi, George

---
<p align="center">
  <img src="https://storage.googleapis.com/kaggle-datasets-images/138/287/229bfb5d3dd1a49cc5ac899c45ca2213/dataset-cover.png"> 
</p>

# 1. Overview of the project and Motivation 
### Motivation: 
To predict the Box office revenue of a movie based on it's characteristics. 
Our analysis will allow Directors/Producers to decide on what characteristics of the movie will affect their box office revenue, and what to modify in their selection of actors or investment in the movies to maximize their profit. Such analysis will also allow other interested third parties to predict the success of a film before it is released.      
We aim to find the variables most associated with film revenue, and to see how the various revenue prediction models are affected by them.

---
# 2. Dataset and visualization 

### (1). Dataset: The Movies DataBase (TMDB) 5000 Movie Dataset (from Kaggle)
#### Features in the dataset: 24 features in total &nbsp;&nbsp;&nbsp;
<!--
1. ID    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. movie_id
3. title &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. cast
5. crew
6. budget
7. genres
8. homepage
9. id
10. keywords
11. original_language
12. original_title
13. overview
14. popularity
15. production_companies
16. production_countries
17. release_date
18. revenue
19. runtime
20. spoken_languages
21. status
22. tagline
23. title
24. vote_average
25. vote_count  
-->
<table>

  <tr>
    <td>ID</td>
    <td>Title</td>
    <td>crew</td>
  </tr>
  <tr>
    <td>budget</td>
    <td>genres</td>
    <td>homepage</td>
  </tr>
  <tr>
    <td>id</td>
    <td>keywords</td>
    <td>original_language</td>
  </tr>
  <tr>
    <td>original_title/td>
    <td>overview</td>
    <td>popularity</td>
  </tr>
  <tr>
    <td>production_companies</td>
    <td>production_countries</td>
    <td>release_date</td>
  </tr>
  <tr>
    <td>revenue</td>
    <td>runtime</td>
    <td>spoken_languages</td>
  </tr>
  <tr>
    <td>status</td>
    <td>tagline</td>
    <td>title</td>
  </tr>
  <tr>
    <td>vote_average</td>
    <td>vote_count</td>
    <td></td>
  </tr>
</table>

#### Visualization: 
Binning movies into Revenue bins of 100 Million
<p align="center">
  <img src="PrithviCodes/RevenueVSCount.png">
</p>


---
# 3. Data pre-processing
#### Steps followed for Data cleaning & Data pre-processing:
- Removal of data points with missing revenues
- Removing zero REVENUES from the data 
- Adjusting revenue for inflation.
- Separation of year into Year and day of the year, since we theorized that film revenue will be highly correlated with which season the movie is released in.
- Encoding categorical features: conversion of data into binary format.
  - Different classes in a column (Lists) allotted their own column, and each row will indicate if column existed or not by assigning either a 1 or a 0. 
- Data was then divided into Test Validation and Training sets (60%, 20% and 20%) for further model training and testing.


---
# 4. Feature Reduction 

Our data has 10,955 features, which is huge, especially in relation to the 3376 data points. To reduce the number of features to increase speed of running supervised learning algorithms for revenue prediction of the movies, feature reduction was deemed required. To achieve this, PCA and feature selection were pursued.
 
### (1). PCA (Sanmesh)
 
PCA was done in two ways:
1. (PCA_noScale_20Comp) Data wasn't scaled, and number of principal components selected = 20
2. (PCA_Scale_0.995VarRecov) Z-Score normalization was done on the features, and number of principal components = # to recover 99% of the variance. To achieve normalization, remove the mean of the feature and scale to unit variance. The Z-Score of a sample x is calculated as: z = (x - u) / s.

#### PCA_noScale_20Comp DETAILS
Recovered Variance: 99.99999999999851  
Original # features: 10955  
Reduced # features: 20  
Recovered Variance Plot Below for PCA_noScale_20Comp%    
Note: Huge first principal component is probably due to othe feature of budget, which is much bigger than all other features (average = 40,137,051.54)  
<p align="center">
  <img src="SanmeshCodes/Figures/20CompPCAGraph.png" >
</p>

#### PCA_Scale_99%VarRecov DETAILS
Recovered Variance:  99.00022645866223  
Original # features: 10955  
Reduced # features: 2965  
Recovered Variance Plot Below for PCA_Scale_99%VarRecov  
<p align="center">
  <img src="SanmeshCodes/Figures/99PercRecovVarPCAGraph.png" >
</p>

### (2). Feature selection (Prithvi)

#### Using XGBRegressor

We used XGBRegressor to check out the correlation of various features to the revenue. 
Once we visualized the graphs we then manually set a threshold and gathered 150 features for testing our models on.

#### Graphs



##### Feature importances of encoded movie data
######  2000 features sorted by feature importance scores of XGBRegressor
<p align="left">
  <img src="PrithviCodes/plots/xgb_2000.png" >
</p>

###### 150 to 200 features feature importance scores of XGBRegressor

To determine threshold for cutoff for feature selection
<p align="left">
  <img src="PrithviCodes/plots/xgb_150_200.png" >
</p>

##### Top 25 Revenue predictors

<p align="center">
  <img src="PrithviCodes/plots/25_top_XGB.png" >
</p>




# 5. Movie Revenue Prediction 

## Experiments and Model Testing

<p align="center">
  <img src="PrithviCodes/plots/flow_chart.png">
</p>

### Linear ridge regression (Sanmesh)

First, we tried to predict the exact revenue of the test set of movies using linear ridge regression. Ridge regression was chosen because it would it would protect against overfitting of the data, especially when there are a huge number of features. 
Cross validation was performed to find the optimal alpha or regularization value.
The data sets were the two PCA data sets, and the feature selection dataset mentioned previously. Ridge Regression was trained on 80% of each data set, and then finally tested on the remaining 20% of the data sets.  The results are below.

### (1). PCA No scaling, 20 components
RMSE: 160266397.7589437  
R2 score 0.49805732362034183  

### (2). PCA Scaling,99% variance recovery:
RMSE: 225957444.3019453  
R2 score 0.00224829444458019  

### (3). Feature Selection:
RMSE: 126001088.6944168  
R2 score 0.6897457309459162  

Comparing RMSE and R2 of Ridge Regression on Three Input Data
<p align="center">
  <img src="SanmeshCodes/Figures/RidgeRegressionRMSE.PNG">
</p>
<p align="center">
  <img src="SanmeshCodes/Figures/RidgeRegressionR%5E2.PNG">
</p>

The plots below are the predicted vs actual revenue predicted from Ridge Regression. The data was sorted by the actual y values in order to make it easier to view the results. Alpha was determined through kfold method and was 0.5 for feature selection.  

Revenue Prediction with PCA_noScale_20Comp data as input
<p align="center">
  <img src="SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtest_PCA20Comp.png">
</p>
  
Revenue Prediction with PCA_Scale_99%VarRecov data as input
<p align="center">
  <img src="SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtest_PCA99PercVarRecov.png">
</p>  
  
Revenue Prediction with Feature Selection data as input
<p align="center">
  <img src="SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtest_xgbFeatures.png">
</p>

Closeup of Revenue Prediction with Feature Selection data as input
<p align="center">
  <img src="SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtestCloseup_xgbFeatures.png">
</p>

### (3). Results (Sanmesh)

##### TABLES ( PLACEHOLDER DATA)

##### Classification 
|                        |          |                    F1 SCORE                                 |                   ACCURACY                                  |
|------------------------|----------|--------------------|--------------------|-------------------|--------------------|--------------------|-------------------|
| Models                 | Features | Bin Size: 300$ (M) | Bin Size: 100$ (M) | Bin Size: 50$ (M) | Bin Size: 300$ (M) | Bin Size: 100$ (M) | Bin Size: 50$ (M) |
| Support Vector Machine |          |                    |                    |                   |                    |                    |                   |
| Support Vector Machine |          |                    |                    |                   |                    |                    |                   |
| Random Forest          |          |                    |                    |                   |                    |                    |                   |
| Random Forest          |          |                    |                    |                   |                    |                    |                   |

##### Regression
| Models           | RMSE | R^2 |
|------------------|:----:|----:|
| Ridge Regression |   3  |   1 |

#####  PLOTS & CHARTS

ToDo


------------

The ranking of the input data that gave the highest R^2 scores and lowest RMSE values from best to worst are:
1. Feature Selection 
2. PCA No scaling, 20 components  
3. PCA Scaling, 99% variance recovery

Feature Selection gave us the best performance for ridge regression. Our target R^2 value to indicate a good model is 0.6 to 0.9, according to this literature [1], and this is acheived only through the feature selection data input with ridge regression. Thus we deam ridge regression model with feature selection input as a success in predicting movie revenue.

We can see that for feature selection input, there is bigger error in prediction for bigger test revenues. The predicted revenue plot does have a similar shape to the actual revenue, this showing that the prediction values are trying to follow the actual values. However, the predicted value is not able to keep up with the increase of the actual revenue. This may be because there is a smaller % of actual revenues that are bigger. This may be corrected by having a bigger dataset to train on than only having 3376 samples. 

It isn't clear completely why feature selection performs better than PCA, but one factor may be that there are some features such as a particular actor name, with values only binary 1 or 0 indicating whether the actor is in the movie. So maybe PCA doesn't work as well on binary value features, and there is conflicting opinion on this online in the ML community. Maybe in the future, we will look into other methods of encoding the feature of actors into numerical data. One potential example is having one feature for all actors, and just encoding the actors into a numerical value from 0 to the # of actors.
  
What is interesting is that the PCA data with normalization performed worse than the PCA without normalization. It is counterintuitive because the goal of PCA is to identify the axis with the most variance, and this goal may be impeded when one feature has much bigger values than other features (in our case, the feature is budget). However, the non-normalized PCA might have performed better because the data captures the budget mainly in the first principal component. We see from our correlation graphs and other literature [1] that budget is one of the leading indicators to predicting movie revenue, so it makes sense that when using PCA data without normalization, it will perform better than pca with normalization. Also, some features being binary might also affect the normalization of the data. Also, the PCA normalization data that captured 99% variance had 2965 size, which is a huge amount of features relative to the 3376 data points, and thus might have led to more innacurate results than the PCA without scaling that had only 20 components.

# 6. Classification Models ()


### Binning of Y values 

### SVM
<p align="center">
  <img src="PrithviCodes/BinVSF1SVM_2.png" >
</p>
<p align="left">
  <img src="PrithviCodes/Scatter_predVSActual.png" style="width:50%" >
</p>

<p align="right">
  <img src="PrithviCodes/Scatter_predVSActual100.png" style="width:50%" >
</p>

<p align="center">
  <img src="PrithviCodes/plots/Line_chart_C_f1score.png" >
</p>
<p align="center">
  <img src="PrithviCodes/plots/Line_chart_gamma_f1score.png" >
</p> 


We have plotted our depicted our SVM classification results for both bin sizes below:
<p align="center">
  <img src="Figures/SVM_300_Norm_ConfusionMat.png" >
</p> 

<p align="left">
  <img src="Figures/SVM_100_Norm_ConfusionMat.png" >
</p> 

Although we have achieved high accuracy and F1-score, we see from the confusion matrix that majority of our test instances are predicted to be in bin 0 or 1. This can be explained by class imbalance in the training data. There are more number of examples which belong to class 0 and 1 as compared to other categories. To overcome this challenge, we explored Random Forest which performs better with class imbalance in training data.

Another possible avenue to explore for this class imbalance problem with SVM would be to increase the penalty for misclassifying minority classes. This could be done by inversely weighing 'C' with class size. This could be explored in future.


### Random Forest
<p align="center">
  <img src="Tarushree_RF plots/F1VsDepth_300.png" height="500" width="600">
</p>
<p align="center">
  <img src="Tarushree_RF plots/F1VsEst_300.png" height="500" width="600">
</p>
<p align="center">
  <img src="Tarushree_RF plots/F1VsDepth_100.png" height="500" width="600">
</p>
<p align="center"> 
  <img src="Tarushree_RF plots/F1VsEst_100.png" height="500" width="600">
</p>
<p align="center">
  <img src="Tarushree_RF plots/Scatter_RF_300.png" height="500" width="600">
</p>
<p align="center">
  <img src="Tarushree_RF plots/Scatter_RF_100.png" height="500" width="600">
</p>
<p align="center">
  <img src="Tarushree_RF plots/BinSize_vs_F1_barplot.png" height="500" width="600">
</p>

<p align="center">
  <img src="Figures/RF_300_Norm_ConfusionMat.png" >
</p>
<p align="left">
  <img src="Figures/RF_100_Norm_ConfusionMat.png" >
</p>




# 7. Final Conclusions


# 8. Reference
[1] What makes a successful film? Predicting a film’s revenue and user rating with machine learning. (2019). Retrieved 28 September 2019, from https://towardsdatascience.com/what-makes-a-successful-film-predicting-afilms-revenue-and-user-rating-with-machine-learning-e2d1b42365e7 
[2] h98uesrhmdifvherdsfredsiojfveioadschj 8aerdsh ciofvjerdiosjcf[mxerwjds9gtkvsed
[3] arkpofaewksporkfweoskrpfksdpokafkewdsofkesdopxrkopfwae,csporkf4	wq[dskxf
