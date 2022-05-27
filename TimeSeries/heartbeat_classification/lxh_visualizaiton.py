import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def draw_pie(datas, names, explode=None, show_boy=None, title=None,ratio=0.02, ax=None, autofig=True, figsize=(8, 6), dpi=100, title_y=1):
    if autofig:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.gca()
    else:
        assert ax is not None
    def myautopct(pct):
        return f'{pct:1.1f}%' if pct > ratio * 100 else ''
    labels = []
    for i, name in enumerate(names):
        if datas[i] > np.sum(datas) * ratio:
            labels.append(name)
        else:
            labels.append('')
    if explode is None:
        explode = [0 if name != show_boy else 0.1 for name in names]
    
    ax.pie(datas, explode=explode, labels=labels, autopct=myautopct,
            shadow=True, startangle=90)
    if title is not None:
        ax.set_title(title, y=title_y)
    if autofig:
        fig.tight_layout()
        plt.show()
        
def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    plt.show()

