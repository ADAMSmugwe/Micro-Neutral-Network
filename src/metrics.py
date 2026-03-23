import numpy as np



def confusion_matrix(y_true, y_pred):
    """Build an (n_classes × n_classes) confusion matrix from integer or one-hot labels."""
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    n_classes = int(max(np.max(y_true), np.max(y_pred))) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report(y_true, y_pred, classes=None):
    """
    Returns a dict mapping each class label to {'precision', 'recall', 'f1', 'support'}.
    Also includes a 'macro avg' and 'weighted avg' entry.
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    report = {}

    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = int(np.sum(cm[i, :]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        key = classes[i] if classes is not None else i
        report[key] = {
            'precision': float(precision),
            'recall':    float(recall),
            'f1':        float(f1),
            'support':   support,
        }

    keys  = list(report.keys())
    total = sum(report[k]['support'] for k in keys)
    macro_p  = np.mean([report[k]['precision'] for k in keys])
    macro_r  = np.mean([report[k]['recall']    for k in keys])
    macro_f1 = np.mean([report[k]['f1']        for k in keys])

    w_p  = sum(report[k]['precision'] * report[k]['support'] for k in keys) / total
    w_r  = sum(report[k]['recall']    * report[k]['support'] for k in keys) / total
    w_f1 = sum(report[k]['f1']        * report[k]['support'] for k in keys) / total

    report['macro avg']    = {'precision': float(macro_p),  'recall': float(macro_r),
                               'f1': float(macro_f1),        'support': total}
    report['weighted avg'] = {'precision': float(w_p),      'recall': float(w_r),
                               'f1': float(w_f1),            'support': total}
    return report


def print_classification_report(report):
    """Pretty-print the dict returned by classification_report."""
    header = f"{'Class':>14}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}"
    print(header)
    print('-' * len(header))
    skip = {'macro avg', 'weighted avg'}
    for key, vals in report.items():
        if key in skip:
            continue
        print(f"  {str(key):>12}  {vals['precision']:>10.4f}  "
              f"{vals['recall']:>8.4f}  {vals['f1']:>8.4f}  {vals['support']:>8d}")
    print('-' * len(header))
    for key in ('macro avg', 'weighted avg'):
        if key in report:
            vals = report[key]
            print(f"  {key:>12}  {vals['precision']:>10.4f}  "
                  f"{vals['recall']:>8.4f}  {vals['f1']:>8.4f}  {vals['support']:>8d}")



def roc_curve(y_true, y_scores, n_thresholds=100):
    """
    Binary ROC curve.

    Parameters
    ----------
    y_true   : 1-D array of 0/1 labels
    y_scores : 1-D array of predicted probabilities for class 1

    Returns
    -------
    fpr, tpr : lists of float
    """
    thresholds = np.linspace(1.0, 0.0, n_thresholds)
    tpr_list, fpr_list = [], []
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return fpr_list, tpr_list


def auc(fpr, tpr):
    """Area under an ROC curve via the trapezoidal rule."""
    _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return float(_trapz(tpr, fpr))


def multiclass_roc(y_true_int, y_scores, n_classes):
    """
    One-vs-rest ROC curves for multiclass problems.

    Returns
    -------
    dict mapping class index → (fpr, tpr, auc_score)
    """
    results = {}
    for c in range(n_classes):
        binary_true   = (y_true_int == c).astype(int)
        binary_scores = y_scores[:, c]
        fpr, tpr = roc_curve(binary_true, binary_scores)
        results[c] = (fpr, tpr, auc(fpr, tpr))
    return results



def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """Plot a labelled confusion-matrix heatmap."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f'Saved: {save_path}')
    plt.show()


def show_misclassified(X_test, y_true, y_pred, n=10, save_path=None):
    """
    Display a grid of misclassified images with true / predicted labels.

    Parameters
    ----------
    X_test  : (N, H, W, C) or (N, H, W) array
    y_true  : integer or one-hot labels
    y_pred  : integer or one-hot predicted labels
    """
    import matplotlib.pyplot as plt

    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    mis_idx = np.where(y_true != y_pred)[0]
    n = min(n, len(mis_idx))
    if n == 0:
        print('No misclassified examples found.')
        return

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))
    if n == 1:
        axes = [axes]
    for i, idx in enumerate(mis_idx[:n]):
        img = X_test[idx].squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'T:{y_true[idx]}\nP:{y_pred[idx]}', fontsize=9)
        axes[i].axis('off')
    plt.suptitle('Misclassified Examples', fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f'Saved: {save_path}')
    plt.show()


def plot_filters(conv_layer, save_path=None):
    """
    Visualise all filters from the first conv layer.
    Expects filters of shape (f, f, in_channels, out_channels).
    Only the first input channel is shown.
    """
    import matplotlib.pyplot as plt

    filters = conv_layer.filters
    n_filters = filters.shape[3]
    ncols = min(n_filters, 8)
    nrows = (n_filters + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if idx < n_filters:
            f = filters[:, :, 0, idx]
            f_min, f_max = f.min(), f.max()
            if f_max > f_min:
                f = (f - f_min) / (f_max - f_min)
            ax.imshow(f, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'F{idx}', fontsize=8)
        ax.axis('off')

    plt.suptitle('Learned Conv1 Filters', fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f'Saved: {save_path}')
    plt.show()


def plot_roc_curves(roc_results, n_classes, save_path=None):
    """Plot one-vs-rest ROC curves for every class."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(n_classes):
        fpr, tpr, auc_score = roc_results[c]
        ax.plot(fpr, tpr, label=f'Class {c} (AUC={auc_score:.3f})', linewidth=1.2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('One-vs-Rest ROC Curves')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f'Saved: {save_path}')
    plt.show()
