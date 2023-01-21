SHAP values (SHapley Additive exPlanations) is a method based on cooperative game theory and used to increase transparency and interpretability of machine learning models.

Linear models, for example, can use their coefficients as a metric for the overall importance of each feature, but they are scaled with the scale of the variable itself, which might lead to distortions and misinterpretations. Also, the coefficient cannot account for the local importance of the feature, and how it changes with lower or higher values. The same can be said for feature importances of tree-based models, and this is why SHAP is useful for interpretability of models.

Important: while SHAP shows the contribution or the importance of each feature on the prediction of the model, it does not evaluate the quality of the prediction itself.

Consider a coooperative game with the same number of players as the name of features. SHAP will disclose the individual contribution of each player (or feature) on the output of the model, for each example or observation.

Given the California Housing Dataset [1,2](available on the scikit-learn library), we can isolate one single observation and calculate the SHAP values for this single data point:

shap.plots.waterfall(shap_values[x])
In the waterfall above, the x-axis has the values of the target (dependent) variable which is the house price. x is the chosen observation, f(x) is the predicted value of the model, given input x and E[f(x)] is the expected value of the target variable, or in other words, the mean of all predictions (mean(model.predict(X))).

The SHAP value for each feature in this observation is given by the length of the bar. In the example above, Longitude has a SHAP value of -0.48, Latitude has a SHAP of +0.25 and so on. The sum of all SHAP values will be equal to E[f(x)] — f(x).

The absolute SHAP value shows us how much a single feature affected the prediction, so Longitude contributed the most, MedInc the second one, AveOccup the third, and Population was the feature with the lowest contribution to the preditcion.

Note that these SHAP values are valid for this observation only. With other data points the SHAP values will change. In order to understand the importance or contribution of the features for the whole dataset, another plot can be used, the bee swarm plot:

shap.plots.beeswarm(shap_values)

For example, high values of the Latitude variable have a high negative contribution on the prediction, while low values have a high positive contribution.

The MedInc variable has a really high positive contribution when its values are high, and a low negative contribution on low values. The feature Population has almost no contribution to the prediction, whether its values are high or low.

All variables are shown in the order of global feature importance, the first one being the most important and the last being the least important one.

Effectively, SHAP can show us both the global contribution by using the feature importances, and the local feature contribution for each instance of the problem by the scattering of the beeswarm plot.

To use SHAP in Python we need to install SHAP module:

pip install shap
