# Machine Learning Metrics

| Sl. No. | Name of the Metrics |
| :---: | :--- |
| 1. | Accuracy |
| 2. | True Positives |
| 3. | True Negatives |
| 4. | False Positives |
| 5. | False Negatives |
| 6. | Confusion Matrix |
| 7. | Binary Accuracy |
| 8. | Multiclass Accuracy |
| 9. | Precision <-> Positive Predictive Value |
| 10. | F beta score |
| 11. | F1score_cm |
| 12. | F2 score (beta=2) |
| 13. | True Positives rate <-> sensitivity <-> recall |
| 14. | True Negatives Rate <-> specificity <-> recall for neg. class |
| 15. | ROC Curve |
| 16. | ROC AUC score |
| 17. | Precision Recall Curve |
| 18. | False Positives rate (Type I Error) |
| 19. | False Negatives Rate (Type II Error) |
| 20. | Negative Predictive Value |
| 21. | False Discovery Rate |
| 22. | Cohen Kappa Metric |
| 23. | Matthews Correlation Coefficient MCC |
| 24. | PR AUC score (Average precision) |
| 25. | Log loss |
| 26. | Brier score |
| 27. | Cumulative gains chart |
| 28. | Lift curve (lift chart) |
| 29. | Kolmogorov-Smirnov plot |
| 30. | Kolmogorov-Smirnov statistic |

## Accuracy

It measures how many observations, both positive and negative, were correctly classified.

You shouldn’t use accuracy on imbalanced problems. Then, it is easy to get a high accuracy score by simply classifying all observations as the majority class. For example in our case, by classifying all transactions as non-fraudulent we can get an accuracy of over 0.9.

### When to use it

- When your problem is balanced using accuracy is usually a good start. An additional benefit is that it is really easy to explain it to non-technical stakeholders in your project,
- When every class is equally important to you.

- - -

## True Positives

A true positive is an outcome where the model correctly predicts the positive class.

- - -

## True Negatives

A true negative is an outcome where the model correctly predicts the negative class.

- - -

## False Positives

A false positive is an outcome where the model incorrectly predicts the positive class.

- - -

## False Negatives

A false negative is an outcome where the model incorrectly predicts the negative class.

- - -

## Confusion Matrix

A confusion matrix is a table that is used to define the performance of a classification algorithm. A confusion matrix visualizes and summarizes the performance of a classification algorithm.

- - -

## Binary Accuracy

Binary Accuracy calculates the percentage of predicted values (yPred) that match with actual values (yTrue) for binary labels. Since the label is binary, yPred consists of the probability value of the predictions being equal to 1.

- - -

## Multiclass Accuracy

Accuracy is one of the most popular metrics in multi-class classification and it is directly computed from the confusion matrix. The formula of the Accuracy considers the sum of True Positive and True Negative elements at the numerator and the sum of all the entries of the confusion matrix at the denominator.

- - -

## Precision <-> Positive Predictive Value

It measures how many observations predicted as positive are in fact positive. Taking our fraud detection example, it tells us what is the ratio of transactions correctly classified as fraudulent.

When you are optimizing precision you want to make sure that people that you put in prison are guilty.

### When to use it

- Again, it usually doesn’t make sense to use it alone but rather coupled with other metrics like recall.
- When raising false alerts is costly, when you want all the positive predictions to be worth looking at you should optimize for precision.

- - -

## F beta score

Simply put, it combines precision and recall into one metric. The higher the score the better our model is. You can calculate it in the following way:

When choosing beta in your F-beta score the more you care about recall over precision the higher beta you should choose. For example, with F1 score we care equally about recall and precision with F2 score, recall is twice as important to us.

With 0<beta<1 we care more about precision and so the higher the threshold the higher the F beta score. When beta>1 our optimal threshold moves toward lower thresholds and with beta=1 it is somewhere in the middle.

- - -

## F1score_cm

(beta=1)
It’s the harmonic mean between precision and recall.

As we can see combining precision and recall gave us a more realistic view of our models. We get 0.808 for the best one and a lot of room for improvement.

What is good is that it seems to be ranking our models correctly with those larger lightGBMs at the top.

We can adjust the threshold to optimize F1 score. Notice that for both precision and recall you could get perfect scores by increasing or decreasing the threshold. Good thing is, you can find a sweet spot for F1metric. As you can see, getting the threshold just right can actually improve your score by a bit 0.8077->0.8121.

### When to use it

- Pretty much in every binary classification problem. It is my go-to metric when working on those problems. It can be easily explained to business stakeholders.

- - -

## F2 score (beta=2)

It’s a metric that combines precision and recall, putting 2x emphasis on recall.

This score is even lower for all the models than F1 but can be increased by adjusting the threshold considerably.Again, it seems to be ranking our models correctly, at least in this simple example.

We can see that with a lower threshold and therefore more true positives recalled we get a higher score. You can usually find a sweet spot for the threshold. Possible gain from 0.755 -> 0.803 show how important threshold adjustments can be here.

### When to use it

- I’d consider using it when recalling positive observations (fraudulent transactions) is more important than being precise about it

- - -

## True Positives rate <-> sensitivity <-> recall

It measures how many observations out of all positive observations have we classified as positive. It tells us how many fraudulent transactions we recalled from all fraudulent transactions.

### When to use it

- Usually, you will not use it alone but rather coupled with other metrics like precision.
- That being said, recall is a go-to metric, when you really care about catching all fraudulent transactions even at a cost of false alerts. Potentially it is cheap for you to process those alerts and very expensive when the transaction goes unseen.

- - -

## True Negatives Rate <-> specificity <-> recall for neg. class

It measures how many observations out of all negative observations have we classified as negative. In our fraud detection example, it tells us how many transactions, out of all non-fraudulent transactions, we marked as clean.

### When to use it

- Usually, you don’t use it alone but rather as an auxiliary metric,
- When you really want to be sure that you are right when you say something is safe. A typical example would be a doctor telling a patient “you are healthy”. Making a mistake here and telling a sick person they are safe and can go home is something you may want to avoid.

- - -

## ROC Curve

It is a chart that visualizes the tradeoff between true positive rate (TPR) and false positive rate (FPR). Basically, for every threshold, we calculate TPR and FPR and plot it on one chart.

Of course, the higher TPR and the lower FPR is for each threshold the better and so classifiers that have curves that are more top-left side are better.

Extensive discussion of ROC Curve and ROC AUC score can be found in this article by Tom Fawcett.

We can see a healthy ROC curve, pushed towards the top-left side both for positive and negative class. It is not clear which one performs better across the board as with FPR < ~0.15 positive class is higher and starting from FPR~0.15 the negative class is above.

- - -

## ROC AUC score

In order to get one number that tells us how good our curve is, we can calculate the Area Under the ROC Curve, or ROC AUC score. The more top-left your curve is the higher the area and hence higher ROC AUC score.

Alternatively, it can be shown that ROC AUC score is equivalent to calculating the rank correlation between predictions and targets. From an interpretation standpoint, it is more useful because it tells us that this metric shows how good at ranking predictions your model is. It tells you what is the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.

We can see improvements and the models that one would guess to be better are indeed scoring higher. Also, the score is independent of the threshold which comes in handy.

### When to use it

- You should use it when you ultimately care about ranking predictions and not necessarily about outputting well-calibrated probabilities (read this article by Jason Brownlee if you want to learn about probability calibration).
- You should not use it when your data is heavily imbalanced. It was discussed extensively in this article by Takaya Saito and Marc Rehmsmeier. The intuition is the following: false positive rate for highly imbalanced datasets is pulled down due to a large number of true negatives.
- You should use it when you care equally about positive and negative classes. It naturally extends the imbalanced data discussion from the last section. If we care about true negatives as much as we care about true positives then it totally makes sense to use ROC AUC.

- - -

## Precision Recall Curve

It is a curve that combines precision (PPV) and Recall (TPR) in a single visualization. For every threshold, you calculate PPV and TPR and plot it. The higher on y-axis your curve is the better your model performance.

You can use this plot to make an educated decision when it comes to the classic precision/recall dilemma. Obviously, the higher the recall the lower the precision. Knowing at which recall your precision starts to fall fast can help you choose the threshold and deliver a better model.

We can see that for the negative class we maintain high precision and high recall almost throughout the entire range of thresholds. For the positive class precision is starting to fall as soon as we are recalling 0.2 of true positives and by the time we hit 0.8, it decreases to around 0.7.

- - -

## False Positives rate (Type I Error)

When we predict something when it isn’t we are contributing to the false positive rate. You can think of it as a fraction of false alerts that will be raised based on your model predictions.

### When to use it

- You rarely would use this metric alone. Usually as an auxiliary one with some other metric,
- If the cost of dealing with an alert is high you should consider increasing the threshold to get fewer alerts.

- - -

## False Negatives Rate (Type II Error)

When we don’t predict something when it is, we are contributing to the false negative rate. You can think of it as a fraction of missed fraudulent transactions that your model lets through.

### When to use it

- Usually, it is not used alone but rather with some other metric,
- If the cost of letting the fraudulent transactions through is high and the value you get from the users isn’t you can consider focusing on this number.

- - -

## Negative Predictive Value

It measures how many predictions out of all negative predictions were correct. You can think of it as precision for negative class. With our example, it tells us what is the fraction of correctly predicted clean transactions in all non-fraudulent predictions.

All models score really high and no wonder, since with an imbalanced problem it is easy to predict negative class.

The higher the threshold the more cases are classified as negative and the score goes down. However, in our imbalanced example even at a very high threshold, the negative predictive value is still good.

### When to use it:

- When we care about high precision on negative predictions. For example, imagine we really don’t want to have any additional process for screening the transactions predicted as clean. In that case, we may want to make sure that our negative predictive value is high.

- - -

## False Discovery Rate

It measures how many predictions out of all positive predictions were incorrect. You can think of it as simply 1-precision. With our example, it tells us what is the fraction of incorrectly predicted fraudulent transactions in all fraudulent predictions.

The “best model” is incredibly shallow lightGBM which we expect to be incorrect (deeper model should work better).

That is an important takeaway, looking at precision (or recall) alone can lead to you selecting a suboptimal model.

The higher the threshold, the less positive predictions. The less positive predictions, the ones that are classified as positive have higher certainty scores. Hence, the false discovery rate goes down.

### When to use it

- Again, it usually doesn’t make sense to use it alone but rather coupled with other metrics like recall.
- When raising false alerts is costly and when you want all the positive predictions to be worth looking at you should optimize for precision.

- - -

## Cohen Kappa Metric

In simple words, Cohen Kappa tells you how much better is your model over the random classifier that predicts based on class frequencies.

To calculate it one needs to calculate two things: “observed agreement” (po) and “expected agreement” (pe). Observed agreement (po) is simply how our classifier predictions agree with the ground truth, which means it is just accuracy. The expected agreement (pe) is how the predictions of the random classifier that samples according to class frequencies agree with the ground truth, or accuracy of the random classifier.

From an interpretation standpoint, I like that it extends something very easy to explain (accuracy) to situations where your dataset is imbalanced by incorporating a baseline (dummy) classifier.

We can easily distinguish the worst/best models based on this metric. Also, we can see that there is still a lot of room to improve our best model.

With the chart just like the one above we can find a threshold that optimizes cohen kappa. In this case, it is at 0.31 giving us some improvement 0.7909 -> 0.7947 from the standard 0.5.

### When to use it

- This metric is not used heavily in the context of classification. Yet it can work really well for imbalanced problems and seems like a great companion/alternative to accuracy.

- - -

## Matthews Correlation Coefficient MCC

It’s a correlation between predicted classes and ground truth. It can be calculated based on values from the confusion matrix:

Alternatively, you could also calculate the correlation between y_true and y_pred.

We can clearly see improvements in our model quality and a lot of room to grow, which I really like. Also, it ranks our models reasonably and puts models that you’d expect to be better on top. Of course, MCC depends on the threshold that we choose.

We can adjust the threshold to optimize MCC. In our case, the best score is at 0.53 but what I really like is that it is not super sensitive to threshold changes.

### When to use it

- When working on imbalanced problems,
- When you want to have something easily interpretable.

- - -

## PR AUC score (Average precision)

Similarly to ROC AUC score you can calculate the Area Under the Precision-Recall Curve to get one number that describes model performance.

You can also think about PR AUC as the average of precision scores calculated for each recall threshold [0.0, 1.0]. You can also adjust this definition to suit your business needs by choosing/clipping recall thresholds if needed.

The models that we suspect to be “truly” better are in fact better in this metric which is definitely a good thing. Overall, we can see high scores but way less optimistic then ROC AUC scores (0.96+).

### When to use it

- when you want to communicate precision/recall decision to other stakeholders
- when you want to choose the threshold that fits the business problem.
- when your data is heavily imbalanced. As mentioned before, it was discussed extensively in this article by Takaya Saito and Marc Rehmsmeier. The intuition is the following: since PR AUC focuses mainly on the positive class (PPV and TPR) it cares less about the frequent negative class.
- when you care more about positive than negative class. If you care more about the positive class and hence PPV and TPR you should go with Precision-Recall curve and PR AUC (average precision).

- - -

## Log loss
Log loss is often used as the objective function that is optimized under the hood of machine learning models. Yet, it can also be used as a performance metric.

Basically, we calculate the difference between ground truth and predicted score for every observation and average those errors over all observations. For one observation the error formula reads:

The more certain our model is that an observation is positive when it is, in fact, positive the lower the error. But this is not a linear relationship. It is good to take a look at how the error changes as that difference increases

So our model gets punished very heavily when we are certain about something that is untrue. For example, when we give a score of 0.9999 to an observation that is negative our loss jumps through the roof. That is why sometimes it makes sense to clip your predictions to decrease the risk of that happening.

It is difficult to really see strong improvement and get an intuitive feeling for how strong the model is. Also, the model that was chosen as the best one before (BIN-101) is in the middle of the pack. That can suggest that using log-loss as a performance metric can be a risky proposition.

### When to use it:

- Pretty much always there is a performance metric that better matches your business problem. Because of that, I would use log-loss as an objective for your model with some other metric to evaluate performance.

- - -

## Brier score

It is a measure of how far your predictions lie from the true values. For one observation it simply reads:

Basically, it is a mean square error in the probability space and because of that, it is usually used to calibrate probabilities of the machine learning models. If you want to read more about probability calibration I recommend that you read this article by Jason Brownlee.

It can be a great supplement to your ROC AUC score and other metrics that focus on other things.

Model from the experiment BIN-101 has the best calibration and for that model, on average our predictions were off by 0.16 (√0.0263309).

### When to use it

- When you care about calibrated probabilities

- - -

## Cumulative gains chart

In simple words, it helps you gauge how much you gain by using your model over a random model for a given fraction of top scored predictions.

Simply put:

- you order your predictions from highest to lowest and
- for every percentile you calculate the fraction of true positive observations up to that percentile.
It makes it easy to see the benefits of using your model to target given groups of users/accounts/transactions especially if you really care about sorting them.

We can see that our cumulative gains chart shoots up very quickly as we increase the sample of highest-scored predictions. By the time we get to the 20th percentile over 90% of positive cases are covered. You could use this chart to prioritize and filter out possible fraudulent transactions for processing. 

Say we were to use our model to assign possible fraudulent transactions for processing and we needed to prioritize. We could use this chart to tell us where it makes the most sense to choose a cutoff.

### When to use it

- Whenever you want to select the most promising customers or transactions to target and you want to use your model for sorting.
- It can be a good addition to ROC AUC score which measures ranking/sorting performance of your model.

- - -

## Lift curve (lift chart)

It is pretty much just a different representation of the cumulative gains chart:

- we order the predictions from highest to lowest
- for every percentile, we calculate the fraction of true positive observations up to that percentile for our model and for the random model,
- we calculate the ratio of those fractions and plot it.

It tells you how much better your model is than a random model for the given percentile of top scored predictions.

So for the top 10% of predictions, our model is over 10x better than random, for 20% is over 4x better and so on.

### When to use it:

- Whenever you want to select the most promising customers or transactions to target and you want to use your model for sorting.
- It can be a good addition to ROC AUC score which measures ranking/sorting performance of your model.

- - -

## Kolmogorov-Smirnov plot

KS plot helps to assess the separation between prediction distributions for positive and negative classes.

In order to create it you:

- sort your observations by the prediction score,
- for every cutoff point [0.0, 1.0] of the sorted dataset (depth) calculate the proportion of true positives and true negatives in this depth,
- plot those fractions, positive(depth)/positive(all), negative(depth)/negative(all), on Y-axis and dataset depth on X-axis.

So it works similarly to cumulative gains chart but instead of just looking at positive class it looks at the separation between positive and negative class.

So we can see that the largest difference is at a cutoff point of 0.034 of top predictions. After that threshold, it decreases at a moderate rate as we increase the percentage of top predictions. Around 0.8 it is really getting worse really fast. So even though the best separation is at 0.034 we could potentially push it a bit higher to get more positively classified observations.

- - -

## Kolmogorov-Smirnov statistic

If we want to take the KS plot and get one number that we can use as a metric we can look at all thresholds (dataset cutoffs) from KS plot and find the one for which the distance (separation) between the distributions of true positive and true negative observations is the highest.

If there is a threshold for which all observations above are truly positive and all observations below are truly negative we get a perfect KS statistic of 1.0.

By using the KS statistic as the metric we were able to rank BIN-101 as the best model which we truly expect to be “truly” best model.

### When to use it

- when your problem is about sorting/prioritizing the most relevant observations and you care equally about positive and negative classes.
- It can be a good addition to ROC AUC score which measures ranking/sorting performance of your model.