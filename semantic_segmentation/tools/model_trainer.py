r"""
Module for training model.
"""
import torch
from metrics import accuracy_sum, evaluate_accuracy
from utils import Timer, Animator, Accumulator


def train_batch(net, X, y, loss, trainer, device):
    r"""
    进行一个batch的训练, 暂时不考虑多GPU训练
    """
    # 暂时不考虑多GPU训练
    X = X.to(device)
    y = y.to(device)
    net.train()
    trainer.zero_grad()
    y_pred = net(X)
    l = loss(y_pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy_sum(y_pred, y)
    return train_loss_sum, train_acc_sum

def train_model(net, train_iter, valid_iter, loss, trainer, num_epochs, device):
    r"""
    进行一次完整的训练
    """
    timer, num_batches = Timer(), len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'valid acc'])
    metric = Accumulator(4)
    net = net.to(device)
    for epoch in range(num_epochs):
        # 存储4个指标：sum of train loss, acc, number of examples, predictions
        metric.reset()
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features, labels, loss, trainer, device)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                (metric[0] / metric[2], metric[1] / metric[3], None))
        valid_acc = evaluate_accuracy(net, valid_iter)
        animator.add(epoch + 1, (None, None, valid_acc))
        if epoch == num_epochs - 1: # 统计一下最终训练效果
            print(f'loss {metric[0] / metric[2]:.3f}, final train acc '
                f'{metric[1] / metric[3]:.3f}, final valid acc {valid_acc:.3f}')
            print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
                f'{device}')

