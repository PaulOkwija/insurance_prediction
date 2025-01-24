from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np

def plot_roc_curve(y_true, y_pred_prob):
    """
    Plots the ROC curve for the model.

    Parameters:
    y_true (array-like): True labels.
    y_pred_prob (array-like): Predicted probabilities.

    Returns:
    matplotlib.axes._subplots.AxesSubplot: The ROC curve plot.
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    return plt.gca()



def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True.
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    classes (list): List of class labels.
    normalize (bool): Whether to normalize the confusion matrix. Default is False.  

    '''

    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    cm_display.plot(cmap=cmap, ax=ax)
    ax.set_title(title)
    plt.savefig("confusion_matrix.png")
    