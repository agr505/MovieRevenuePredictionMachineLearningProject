# Project title: House Price Analysis and Prediction with Machine Learning Methods
## Team members: Yandong Luo, Xiaochen Peng, Hongwu Jiang and Panni Wang

---
<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Housing_expenses.png"> 
</p>

# 1. Motivaton and overview of the project

---
# 2. Dataset and visulization 

### (1). Dataset: House Sales in King County (from Kaggle)
#### Features in the dataset: 21 features in total
1. id: notation for a house  
2. date: date house was sold  

### (2). dataset visulization
#### Feature distribution
15 features are visulized as below. The features has the following characteristics: 
1. The scale of each feature is quite different, which means normalization is needed
2. There are continuos variables (sqrf_lot et.al), dicreste variables (bedrooms) and categorical variable (grade,yr_renovated)

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Feature_Dist.PNG">
</p>

---
# 3. Data pre-processing
After the dataset is visulized and examined, the data is processed in following ways: 
1. remove irrelevant features: id, date, lat, long, zipcode
2. remove the feature "waterfront" as it is 0 for all the data points
3. normalize the all the features with its mean and sigma as their scale is quite different
4. There is a categorical feature: yr_renovated. It is either 0 or the year that it has been renovated. It is treat as a dummy variable with only two values: "1" if the house has been renovated and "0" if it has not. 

---
# 4. Feature Reduction

To reduce the number of features to increase speed of running supervised learning algorithms for revenue prediction of the movies, feature reduction was deemed required, especially when there are 10955 vs only 3377 data points. To achieve this, PCA and feature selection were pursued.
 
### (1). PCA
 
PCA was done in two ways:
1. (PCA20) No scaling of the data, and picking 20 components
2. (PCA99%) Z-Score normalization of features, and then get the number of components required to recover 99% of the variance. To achieve normalization, remove the mean and scale to unit variance. The standard score of a sample x is calculated as: z = (x - u) / s.

#### PCA20 DETAILS
Recovered Variance: 99.99999999999851
Original # features: 10955
Reduced # features: 20
Recovered Variance Plot for PCA99%
Note: Huge first principal component probably because one of the variables is budget, which is much bigger than all other features
<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/20CompPCAGraph.png" width="400"/>
</p>

#### PCA99% DETAILS
Recovered Variance:  99.00022645866223
Original # features: 10955
Reduced # features: 2965
Recovered Variance Plot for PCA99%
<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/99PercRecovVarPCAGraph.png" width="400"/>
</p>

### (2). Feature selection 

# 5. Movie Revenue Prediction with linear ridge regression

Ridge regression was performed

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
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/RidgeRegressionRMSE.PNG" width="600"/>
</p>
<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/RidgeRegressionR%5E2.PNG" width="600"/>
</p>

Plot below is the predicted vs actual revenue predicted from Ridge Regression with Feature Selection data as input. Alpha was determined through kfold method and was 0.5 for feature selection.
<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtest_xgbFeatures.png" width="400"/>
</p>

Closeup
<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtestCloseup_xgbFeatures.png" width="400"/>
</p>


As the figure shown below, where red line is the real price value, and the blue dots are the predicted price value. The first row shows the linear, lasso and ridge regression without polynomial, the second row shows when polynomial in introduced with degree equals to 2, and the third is with degree equals to 3. It shows that, with polynomial, the prediction achieves better performance, since it can help to fit in non-linear features.

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Predict_All.PNG">
</p>

### (2). Selected Top-10 Features Included

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Predict.PNG">
</p>

The figure shown above is the relation between real price and predicted price, when we only introduced the top-10 important features. The first column shows the linear, ridge and lasso regression, and the second column shows the ones with polynomial (degree is set to 2). Similarly as what we have found in the "all features included" method, the linear regression achieves best performance among all the three linear models.

### (3). Comparison and Discussion

It is shown that Lasso and Ridge regression shows lower RMSE, which indicates more accurate prediction due to less over-fitting. Besides, polynomial regression with 2nd order features shows the lower RMSE loss.  

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/RMSE.PNG" width="400"/>
</p>

Selecting all the features for regression shows slightly lower RMSE than select 10 features. It can be explained that the number of features in this dataset is small (21 in total) and therefore there is no over-fitting by using all the features. 

# 6. Housing price prediction with neural netwok

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/MLP.png" width="300"/>
</p>

## neural network vs linear regression
A 2-layer neural network with fully connected layer is implemented for house price prediction. The hidden layer unit is 64, the activation function at the hidden layer is ReLU and the output is the house price. The prediction is evaluated with root-mean-squred-error (RMSE) of the predicted house price. The neural network is trained with 20 epoch.  
First, the RMSE obtained by neural network method is compared with that of linear regression, as shown in the figure below. Neural network shows lower loss than all the linear regression based methods, which indicates that it can be a good model for house price prediction.

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/MRSE_ANN.png" width="400"/>
</p>

## prediction loss vs. number of hidden units
Then we examined the prediction loss with different neural network settings. The prediction loss of the neural network can be decreased by increasing the number of hidden units, as shown in the figure below. It means that a more complicated model is desired for this task

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Loss_vs_hidden_units.png" width="400"/>
</p>

## prediction loss vs. activation function
Three different activation functions are examined ReLU, Sigmoid and Tanh. The results shows that prediction loss is small when "ReLU" is used while the prediction loss is large when the activation is Sigmoid or Tanh. The prediction loss vs. training epoch is plotted. The neural networks with Sigmoid and Tanh shows slow training as the neuron activation value is limited, which is (0,1) for Sigmoid and (-1,1) for Tanh

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/MRSE_ANN_activations.png" width="300"/> <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Loss_vs_activation_type" width="300"/>
</p>

## Prediction loss vs. optimizers
The prediction loss of neural network trained with different optimization method are also examined with SGD and RMSprop. RMSprop shows faster convergence and less fluctuations when it is convergent because the learning rate can be varied during the training. 

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Loss_vs_optimizer.png" width="400"/>
</p>
  
# 7. Housing recommendation with K-NN
The house recommendation is conducted with k-neareast neighbor algorithm to find the house that best matches the consumer's preference, which is measured by the Euclidean distance between the house in the dataset and the preference input by consumer. An example is shown in the table below, where 5 recommendations are made. It is noted that house price is an important factor as recommendations are trying to match the price expected by consumers. As consumers are price sensitive, it indicates the K-NN works well for house recommendation. 

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Recommendation.JPG">
<p/>

# 8. Discussions (the questions in proposal) 
a. Do all the feature ranking methods list the same informative features? And do those features ranked in the same order?  
Answer: No, there are slight difference. In both recursive feature elimination (RFE), the number of rooms has higher importance while in random forest feature ranking, the area of the rooms has higher importance. However, categorical features such as grade (the grad evaluated by agency) are ranked high by both method

b. With the same set of features, which regression model provides the most accurate prediction.  
Answer: Lasso and Ridge regression with polynomial features (degree = 2) provides the most accurate prediction results as it prevents overfitting. 

c. How to choose the proper methods for prediction  
Answer: in this project, neural network shows the smallest rmse loss for prediction. The factors that influence the prediction accuracy are the number of hidden units, activation functions. For house price prediction, hidden units of larger than 64 is preferred and ReLU activation provides faster training and better accuracy

d. Why not use PCA for feature selection  
Answer: Fisrt, PCA is trying to find the feature or dimension with the highest variance. However, in this project, what we would like to find are the features with the highest impact on house price. Features with high variance may not necessarily have high impact on house price. Besides, PCA may create new features on a new dimension, which is not intepretable.  

# 9. Conclusion
a. Obtained features that influence house price the most  
> Obtained the features that has the highest impact on house price with two feature selection methods: recursive feature elimination (RFE) and random forest.  
> It can be concluded that categorical featuress such as the grade of the house has high impact on house price.  
> Besides, different feature selection method can leads to different results. E.g. for RFE, the number of rooms have the highest impact on house price while the room area is important for house price from random forest based feature selection.  

b. Build the house prediction model
> Both linear regression and neural network are implemented.  
> Neural network provides better prediction.  
> More hidden units and use 'ReLU' as activation can help improve the prediction  

c. House recommendation by K-NN  
> Recommend house based on consumer's needs  
> Price is an important feature to match  

# 10. Reference
[1]Park, B. and J. K. Bae (2015). "Using machine learning algorithms for housing price prediction: The case of Fairfax County, Virginia housing data." Expert Systems with Applications 42(6): 2928-2934.  
[2]Gür Ali, Ö. et. al (2013). "Selecting rows and columns for training support vector regression models with large retail datasets." European Journal of Operational Research 226(3): 471-480.  
[3]Breiman, L. (2001). "Random Forests." Machine Learning 45(1): 5-32.

