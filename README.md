# Project title: House Price Analysis and Prediction with Machine Learning Methods
## Team members: Yandong Luo, Xiaochen Peng, Hongwu Jiang and Panni Wang

---
<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Housing_expenses.png"> 
</p>

# 1. Overview of the project and Motivation (Not Assigned)

---
# 2. Dataset and visualization 

### (1). Dataset: House Sales in King County (from Kaggle) (Not Assigned)
#### Features in the dataset: 21 features in total
1. id: notation for a house  
2. date: date house was sold  

### (2). (Optional) dataset visualization (Not Assigned)
#### Feature distribution

---
# 3. Data pre-processing (Not Assigned)


---
# 4. Feature Reduction 

Our data has 10955 features, which is huge, especially in relation to the 3376 data points. To reduce the number of features to increase speed of running supervised learning algorithms for revenue prediction of the movies, feature reduction was deemed required. To achieve this, PCA and feature selection were pursued.
 
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
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/20CompPCAGraph.png" width="400"/>
</p>

#### PCA_Scale_99%VarRecov DETAILS
Recovered Variance:  99.00022645866223  
Original # features: 10955  
Reduced # features: 2965  
Recovered Variance Plot Below for PCA_Scale_99%VarRecov  
<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/99PercRecovVarPCAGraph.png" width="400"/>
</p>

### (2). Feature selection (Prithvi)

#### Using XGBRegressor
Feature importances of encoded movie data

<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/PrithviCodes/FeatureImportance.png" width="400"/>
</p>

Top 20 Revenue predictors

<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/PrithviCodes/RevenuePredictors20.png" width="400"/>
</p>


# 5. Movie Revenue Prediction with linear ridge regression (Sanmesh)

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
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/RidgeRegressionRMSE.PNG" width="600"/>
</p>
<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/RidgeRegressionR%5E2.PNG" width="600"/>
</p>

The plots below are the predicted vs actual revenue predicted from Ridge Regression. The data was sorted by the actual y values in order to make it easier to view the results. Alpha was determined through kfold method and was 0.5 for feature selection.  

Revenue Prediction with PCA_noScale_20Comp data as input
<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtest_PCA20Comp.png" width="600"/>
</p>
  
Revenue Prediction with PCA_Scale_99%VarRecov data as input
<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtest_PCA99PercVarRecov.png" width="600"/>
</p>  
  
Revenue Prediction with Feature Selection data as input
<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtest_xgbFeatures.png" width="600"/>
</p>

Closeup of Revenue Prediction with Feature Selection data as input
<p align="center">
  <img src="https://github.com/agr505/MovieRevenuePredictionMachineLearningProject/blob/master/SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtestCloseup_xgbFeatures.png" width="600"/>
</p>

### (3). Results (Sanmesh)

The ranking of the input data that gave the highest R^2 scores and lowest RMSE values from best to worst are:
1. Feature Selection 
2. PCA No scaling, 20 components  
3. PCA Scaling,99% variance recovery

Feature Selection gave us the best performance for ridge regression. Our target R^2 value to indicate a good model is 0.6 to 0.9, according to this literature (https://towardsdatascience.com/what-makes-a-successful-film-predicting-a-films-revenue-and-user-rating-with-machine-learning-e2d1b42365e7), and this is acheived only through the feature selection data input with ridge regression. Thus we deam ridge regression model with feature selection input as a success in predicting movie revenue.

We can see that for feature selection input, there is bigger error in prediction for bigger test revenues. The predicted revenue plot does have a similar shape to the actual revenue, this showing that the prediction values are trying to follow the actual values. However, the predicted value is not able to keep up with the increase of the actual revenue. This may be because there is a smaller % of actual revenues that are bigger. This may be corrected by having a bigger dataset to train on than only having 3376 samples. 

It isn't clear completely why feature selection performs better than PCA, but one factor may be that some features are just an actor name, with values only binary 1 or 0 indicating whether the actor is in the movie. So maybe because there are very limited values, it is better to get rid of some of these features through feature selection, rather than transforming these features into new components through PCA. Maybe in the future, we will look into other methods of encoding the feature of actors into numerical data. One potential example is having one feature for all actors, and just encoding the actors into a numerical value from 0 to the # of actors.
  
What is interesting is that the PCA data with normalization performed worse than the PCA without normalization. It is counterintuitive because the goal of PCA is to identify the axis with the most variance, and this goal may be impeded when one feature has much bigger values than other features (in our case, the feature is budget). However, the non-normalized PCA might have performed better because the data captures the budget mainly in the first principal component. We see from our correlation graphs and other literature (https://towardsdatascience.com/what-makes-a-successful-film-predicting-a-films-revenue-and-user-rating-with-machine-learning-e2d1b42365e7) that budget is one of the leading indicators to predicting movie revenue, so it makes sense that when using PCA data without normalization, it will perform better than pca with normalization. 

# 6. Classification Models ()

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

