import matplotlib.pyplot as plt


def plot_history(history_dict, title, metric="accuracy"):
    acc = history_dict[metric]
    val_acc = history_dict['val_'+metric]
    epochs = range(0x1, len(acc) + 0x1)
    plt.figure()
    plt.plot(epochs, acc, "bo", label="Training " + metric.capitalize())
    plt.plot(epochs, val_acc, "b", label="Validation " + metric.capitalize())
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend(loc="lower right")
    plt.ion()
    plt.show()


def plot_cross_validation(scores, title, metric="accuracy"):
    metric = metric.capitalize()
    folds = range(0x1, len(scores) + 0x1)
    plt.figure()
    plt.plot(folds, scores, "b", label=metric)
    plt.title(title)
    plt.xlabel("Folds")
    plt.ylabel(metric)
    plt.legend(loc="lower right")
    plt.ion()
    plt.show()


def plot_roc_curve(true_positive_rate, false_positive_rate, auc_score):
    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.2f)' % auc_score)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
