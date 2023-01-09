"""
1. Accuracy
2. True Positives
3. True Negatives
4. False Positives
5. False Negatives
6. Confusion Matrix
7. Binary Accuracy
8. Multiclass Accuracy
9. Precision <-> Positive Predictive Value
10. F beta score
11. F1score_cm
12. F2 score (beta=2)
13. True Positives rate <-> sensitivity <-> recall
14. True Negatives Rate <-> specificity <-> recall for neg. class
15. ROC Curve
16. ROC AUC score
17. Precision Recall Curve
18. False Positives rate (Type I Error)
19. False Negatives Rate (Type II Error)
20. Negative Predictive Value
21. False Discovery Rate
22. Cohen Kappa Metric
23. Matthews Correlation Coefficient MCC
24. PR AUC score (Average precision)
25. Log loss
26. Brier score
27. Cumulative gains chart
28. Lift curve (lift chart)
29. Kolmogorov-Smirnov plot
30. Kolmogorov-Smirnov statistic
"""

import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, recall_score
from scikitplot.metrics import plot_roc
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, roc_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from scikitplot.metrics import plot_cumulative_gain
from scikitplot.metrics import plot_lift_curve
from scikitplot.metrics import plot_ks_statistic
from scikitplot.helpers import binary_ks_curve



def accuracy_cm(cm):
    return np.trace(cm)/np.sum(cm)

def true_positives(y_true, y_pred):
    tp = 0
    for label, pred in zip(y_true, y_pred):
        if pred == 1 and label == 1:
            tp += 1
    return tp


def true_negatives(y_true, y_pred):
    tn = 0
    for label, pred in zip(y_true, y_pred):
        if pred == 0 and label == 0:
            tn += 1
    return tn


def false_positives(y_true, y_pred):
    fp = 0
    for label, pred in zip(y_true, y_pred):
        if pred == 1 and label == 0:
            fp += 1
    return fp


def false_negatives(y_true, y_pred):
    fn = 0
    for label, pred in zip(y_true, y_pred):
        if pred == 0 and label == 1:
            fn += 1
    return fn

def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert y_true.shape == y_pred.shape
    unique_classes = np.unique(np.concatenate([y_true, y_pred], axis=0)).shape[0]
    cm = np.zeros((unique_classes, unique_classes), dtype=np.int64)

    for label, pred in zip(y_true, y_pred):
        cm[label, pred] += 1

    return cm

def confusion_matrix(y_true, y_pred):
    y_pred_class = y_pred_pos > threshold
    cm = confusion_matrix(y_true, y_pred_class)
    tn, fp, fn, tp = cm.ravel()

    return tn,fp,fn,tp

def binary_accuracy(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    tn = true_negatives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)

def multiclass_accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for label, pred in zip(y_true, y_pred):
        correct += label == pred
    return correct/total

def precision(y_true, y_pred):
    """
    Fraction of True Positive Elements divided by total number of positive predicted units
    How I view it: Assuming we say someone has cancer: how often are we correct?
    It tells us how much we can trust the model when it predicts an individual as positive.
    """
    tp = true_negatives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    return tp / (tp + fp)

def precision_cm(cm, average="specific", class_label=1, eps=1e-12):
    tp = np.diagonal(cm)
    fp = np.sum(cm, axis=0) - tp
    #precisions = np.diagonal(cm)/np.maximum(np.sum(cm, axis=0), 1e-12)

    if average == "none":
        return tp/(tp+fp+eps)

    if average == "specific":
        precisions = tp / (tp + fp + eps)
        return precisions[class_label]

    if average == "micro":
        # all samples equally contribute to the average,
        # hence there is a distinction between highly
        # and poorly populated classes
        return np.sum(tp) / (np.sum(tp) + np.sum(fp) + eps)

    if average == "macro":
        # all classes equally contribute to the average,
        # no distinction between highly and poorly populated classes.
        precisions = tp / (tp + fp + eps)
        return np.sum(precisions)/precisions.shape[0]

    if average == "weighted":
        pass


def f_beta_score_cm(cm, average="specific", class_label=1):
    y_pred_class = y_pred_pos > threshold
    fbeta = fbeta_score(y_true, y_pred_class, beta)

    # log score to neptune
    run["logs/fbeta_score"] = fbeta
    return fbeta

def f1score_cm_inbuilt(cm, average="specific", class_label=1):
    y_pred_class = y_pred_pos > threshold
    f1= f1_score(y_true, y_pred_class)

    # log score to neptune
    run["logs/f1_score"] = f1
    return f1

def f1score_cm(cm, average="specific", class_label=1):
    precision = precision_cm(cm, average, class_label)
    recall = recall_cm(cm, average, class_label)
    return 2 * (precision*recall)/(precision+recall)

def f2score_cm(cm, average="specific", class_label=1):
    y_pred_class = y_pred_pos > threshold
    f2 = fbeta_score(y_true, y_pred_class, beta = 2)

    # log score to neptune
    run["logs/f2_score"] = f2
    return f2

# true positive rate <-> sensitivity <-> recall

def recall(y_true, y_pred):
    """
    Recall meaasure the model's predictive accuracy for the positive class.
    How I view it, out of all the people that has cancer: how often are
    we able to detect it?
    """
    tp = true_negatives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    return tp / (tp + fn)

def recall_cm(cm, average="specific", class_label=1, eps=1e-12):
    tp = np.diagonal(cm)
    fn = np.sum(cm, axis=1) - tp

    if average == "none":
        return tp / (tp + fn + eps)

    if average == "specific":
        recalls = tp / (tp + fn + eps)
        return recalls[class_label]

    if average == "micro":
        return np.sum(tp) / (np.sum(tp) + np.sum(fn))

    if average == "macro":
        recalls = tp / (tp + fn + eps)
        return np.sum(recalls)/recalls.shape[0]

    if average == "weighted":
        pass

def recall_inbuilt(y_true, y_pred):
    y_pred_class = y_pred_pos > threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    recall = recall_score(y_true, y_pred_class) # or optionally tp / (tp + fn)

    # log score to neptune
    run["logs/recall_score"] = recall
    return recall

# true negative rate <-> specificity <-> recall for neg. class
def true_negative_rate(y_true, y_pred):
    y_pred_class = y_pred_pos > threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    true_negative_rate = tn / (tn + fp)
    # log score to neptune
    run["logs/true_negative_rate"] = true_negative_rate
    return true_negative_rate

# ROC curve

def roc_curve(y_true, y_preds, plot_graph=True, calculate_AUC=True, threshold_step=0.01):
    TPR, FPR = [], []

    for threshold in np.arange(np.min(y_preds), np.max(y_preds), threshold_step):
        predictions = (y_preds > threshold) * 1
        cm = confusion_matrix(y_true, predictions)
        recalls = recall_cm(cm, average="none")
        # note TPR == sensitivity == recall
        tpr = recalls[1]
        # note tnr == specificity (which is same as recall for the negative class)
        tnr = recalls[0]
        TPR.append(tpr)
        FPR.append(1-tnr)

    if plot_graph:
        plt.plot(FPR, TPR)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.show()

    if calculate_AUC:
        print(np.abs(np.trapz(TPR, FPR)))

def roc_curve_inbuilt(y_true, y_preds, plot_graph=True, calculate_AUC=True, threshold_step=0.01):
    fig, ax = plt.subplots()
    plot_roc(y_true, y_pred, ax=ax)

    # log figure to neptune
    run["images/ROC"].upload(neptune.types.File.as_image(fig))
    return True

# ROC AUC score

def roc_auc_curve(y_true, y_preds, plot_graph=True, calculate_AUC=True, threshold_step=0.01):
    roc_auc = roc_auc_score(y_true, y_pred_pos)

    # log score to neptune
    run["logs/roc_auc_score"] = roc_auc

# Precision Recall Curve
def precision_recall_curve(y_true, y_preds, plot_graph=True, calculate_AUC=True, threshold_step=0.01):
    recalls, precisions = [], []

    for threshold in np.arange(np.min(y_preds), np.max(y_preds), threshold_step):
        predictions = (y_preds > threshold) * 1
        cm = confusion_matrix(y_true, predictions)
        recall = recall_cm(cm, average="specific", class_label=1)
        precision = precision_cm(cm, average="specific", class_label=1)
        recalls.append(recall)
        precisions.append(precision)

    recalls.append(0)
    precisions.append(1)

    if plot_graph:
        plt.plot(recalls, precisions)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.show()

    if calculate_AUC:
        print(np.abs(np.trapz(precisions, recalls)))

# False Positives rate (Type I Error)

def plot_model_boundaries(model, xmin=0, xmax=1.5, ymin=0, ymax=1.5, npoints=40):
    xx = np.linspace(xmin, xmax, npoints)
    yy = np.linspace(ymin, ymax, npoints)
    xv, yv = np.meshgrid(xx, yy)
    xv, yv = xv.flatten(), yv.flatten()
    labels = model.predict(np.c_[xv,yv])
    plt.scatter(xv[labels==1],yv[labels==1],color='r', alpha=0.02, marker='o', s=300)
    plt.scatter(xv[labels==0],yv[labels==0],color='b', alpha=0.02, marker='o', s=300)
    plt.ylim([xmin, xmax])
    plt.xlim([ymin, ymax])

def false_positives_rate():
    np.random.seed(1)
    x = np.concatenate([np.zeros((8, 2)), np.zeros((1,2)), np.zeros((1,2))]) + 0.8*np.random.rand(10,2)
    x[-1,0]+=1
    x[-1,-1]+=1
    y = np.concatenate([np.ones(9), np.zeros(1)])
    model = LinearSVC()
    model.fit(x,y)
    predicted_labels = model.predict(x)
    plot_model_boundaries(model, xmin=0, xmax=1.5, ymin=0, ymax=1.5)
    plt.scatter(x[y==0,0],x[y==0,1], color='b')
    plt.scatter(x[y==1,0],x[y==1,1], color='r');
    fpr, tpr, thresholds = roc_curve(y, predicted_labels)
    plt.title("precision = {}, recall = {}, fpr = {}, tpr = {}".format(precision_score(y, predicted_labels), recall_score(y, predicted_labels), fpr[0], tpr[0]));
    return fpr, tpr, thresholds

# False Negatives Rate (Type II Error)

def false_negative_rate(y_true, y_preds):
    y_pred_class = y_pred_pos > threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    false_negative_rate = fn / (tp + fn)

    # log score to neptune
    run["logs/false_negative_rate"] = false_negative_rate
    return false_negative_rate

# Negative Predictive Value

def negative_predictive_value(y_true, y_preds):
    y_pred_class = y_pred_pos > threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    negative_predictive_value = tn/ (tn + fn)

    # log score to neptune
    run["logs/negative_predictive_value"] = negative_predictive_value
    return negative_predictive_value

# False Discovery Rate

def false_discovery_rate(y_true, y_preds):
    y_pred_class = y_pred_pos > threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    false_discovery_rate = fp/ (tp + fp)

    # log score to neptune
    run["logs/false_discovery_rate"] = false_discovery_rate
    return false_discovery_rate

# Cohen Kappa Metric

def cohen_kappa_metric(y_true, y_pred_class):
    cohen_kappa = cohen_kappa_score(y_true, y_pred_class)
    # log score to neptune
    run["logs/cohen_kappa_score"] = cohen_kappa
    return cohen_kappa

# Matthews Correlation Coefficient MCC

def matthews_correlation_coefficient(y_true, y_pred_class):
    y_pred_class = y_pred_pos > threshold
    matthews_corr = matthews_corrcoef(y_true, y_pred_class)
    run["logs/matthews_corrcoef"] = matthews_corr
    return matthews_corr

# PR AUC score (Average precision)

def PR_AUC_score(y_true, y_pred_pos):
    avg_precision = average_precision_score(y_true, y_pred_pos)

    # log score to neptune
    run["logs/average_precision_score"] = avg_precision
    return avg_precision

# Log loss

def log_loss(y_true, y_pred):
    loss = log_loss(y_true, y_pred)
    # log score to neptune
    run["logs/log_loss"] = loss
    return loss

# Brier score

def brier_score():
    brier_loss = brier_score_loss(y_true, y_pred_pos)
    # log score to neptune
    run["logs/brier_score_loss"] = brier_loss
    return brier_loss

# Cumulative gains chart

def cumulative_gains_chart(y_true, y_pred):
    fig, ax = plt.subplots()
    plot_cumulative_gain(y_true, y_pred, ax=ax)

    # log figure to neptune
    run["images/cumulative_gains"].upload(neptune.types.File.as_image(fig))
    return True

# Lift curve (lift chart)

def lift_curve():
    fig, ax = plt.subplots()
    plot_lift_curve(y_true, y_pred, ax=ax)

    # log figure to neptune
    run["images/lift_curve"].upload(neptune.types.File.as_image(fig))
    return True

# Kolmogorov-Smirnov plot

def kolmogorov_smirnov_plot(y_true, y_pred):
    fig, ax = plt.subplots()
    plot_ks_statistic(y_true, y_pred, ax=ax)

    # log figure to neptune
    run["images/kolmogorov-smirnov"].upload(neptune.types.File.as_image(fig))
    return True

# Kolmogorov-Smirnov statistic

def kolmogorov_smirnov_statistic():
    res = binary_ks_curve(y_true, y_pred_pos)
    ks_stat = res[3]

    # log score to neptune
    run["logs/ks_statistic"] = ks_stat
    return ks_stat

# end def
def balanced_accuracy_cm(cm):
    correctly_classified = np.diagonal(cm)
    rows_sum = np.sum(cm, axis=1)
    indices = np.nonzero(rows_sum)[0]
    if rows_sum.shape[0] != indices.shape[0]:
        warnings.warn("y_pred contains classes not in y_true")
    accuracy_per_class = correctly_classified[indices]/(rows_sum[indices])
    return np.sum(accuracy_per_class)/accuracy_per_class.shape[0]

y = []
probs = []
with open("data.txt") as f:
    for line in f.readlines():
        label, pred = line.split()
        label = int(label)
        pred = float(pred)
        y.append(label)
        probs.append(pred)

precision_recall_curve(y, probs, threshold_step=0.001)
#from sklearn.metrics import precision_recall_curve
#precisions, recalls, _ = precision_recall_curve(y, probs)
#plt.plot(recalls, precisions)
#plt.xlabel("Recall")
#plt.ylabel("Precision")
#plt.title("Precision-Recall curve")
#plt.show()
#print(np.abs(np.trapz(precisions, recalls)))