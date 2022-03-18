r"""
Module for training model.
"""

from tools.model_tester import test_model
from tools.utils import Timer, Animator, Accumulator
from tools.metrics import check_metrics

def train_batch(net, X, y, loss_fn, metrics, optimizer, device):
    r"""
    进行一个batch的训练, 暂时不考虑多GPU训练
    """
    # 暂时不考虑多GPU训练
    X = X.to(device)
    y = y.to(device)
    net.train()
    optimizer.zero_grad()
    y_pred = net(X)
    l = loss_fn(y_pred, y)
    l.backward()
    optimizer.step()
    train_loss_sum = l.sum()
    train_metrics_sum = []
    for metric in metrics:
        train_metrics_sum.append(metric(y_pred, y, method='sum'))
    return train_loss_sum, train_metrics_sum

def train_model(net, train_iter, valid_iter, loss_fn, metrics, optimizer, num_epochs, device, use_tensorboard=False):
    r"""
    进行一次完整的训练 
    """
    # check metrics
    metrics = check_metrics(metrics)
    if use_tensorboard:
        raise NotImplementedError  # TODO 添加tensorboard支持
        train_model_with_Tensorboard(
            net, train_iter, valid_iter, loss_fn, metrics, optimizer, num_epochs, device)
    else:
        train_model_with_Animator(net, train_iter, valid_iter, loss_fn, metrics, optimizer, num_epochs, device)
  

def train_model_with_Animator(net, train_iter, valid_iter, loss_fn, metrics, optimizer, num_epochs, device):
    net = net.to(device)
    print(f"training on {device}")
    num_batches = len(train_iter)
    n_columns = 1 + len(metrics)
    animator = Animator(legends=[['train loss', 'valid loss'],
                        *[[f'train {metric.name}', f'valid {metric.name}'] for metric in metrics]],
                        nrows=1, ncols=n_columns, figsize=(8*n_columns, 8),
                        xlabels=['epoch' for _ in range(n_columns)],
                        ylabels=['loss', *[metric.name for metric in metrics]],
                        xlims=[[0, num_epochs] for _ in range(n_columns)],
                        ylims=None)
    recorder = Accumulator(1 + n_columns, ["num_exaples", "loss", *[f'{metric.name}' for metric in metrics]])
    timer = Timer()
    for epoch in range(num_epochs):
        # 每个epoch开始时重置recorder
        recorder.reset()
        if epoch != 0:
            time_per_epoch = timer.sum() / epoch
        else:
            time_per_epoch = -1
        animator.set_suptitle(
            f'Epoch: {epoch+1}    Device: {device} \nTime already spent: {timer.sum():.3f} sec\nAvg time spent:  {time_per_epoch:.3f} sec/epoch  {timer.avg(): .3f} sec/batch')
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            loss_sum, metrics_sum = train_batch(net, features, labels,
                                 loss_fn, metrics, optimizer, device)
            recorder.add(labels.shape[0], loss_sum, *metrics_sum)
            timer.stop()
            animator.set_suptitle(
                f'Epoch: {epoch+1}    Device: {device} \nTime already spent: {timer.sum():.3f} sec\nAvg time spent:  {time_per_epoch:.3f} sec/epoch  {timer.avg(): .3f} sec/batch')
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(1, 1, epoch + (i + 1) / num_batches,
                            (recorder["loss"] / (i + 1), None))
                for j in range(2, n_columns+1):
                    animator.add(1, j, epoch + (i + 1) / num_batches,
                                ([recorder[metrics[j-2].name] / recorder["num_exaples"], None]))      
        valid_loss, valid_metrics = test_model(net, valid_iter, loss_fn, metrics, device) 
        animator.add(1, 1, epoch + 1, (None, valid_loss))
        for j in range(2, n_columns+1):
            animator.add(1, j, epoch + 1,
                        (None, valid_metrics[j-2]))

    # 统计一下最终训练效果
    print("Train result")
    print(f'loss {recorder["loss"] / num_batches:.3f}')
    for i, metric in enumerate(metrics):
        print(f'final train {metric.name} {recorder[metric.name] / recorder["num_exaples"]:.3f}')
        print(f'final valid {metric.name} {valid_metrics[i]:.3f}')

    print(
        f'{recorder["num_exaples"] * num_epochs / timer.sum():.3f} examples/sec on {device}')
    print(f'{timer.sum() / num_epochs:.3f} sec/epoch on {device}')
    animator.set_suptitle(
        f'Epoch: {num_epochs}    Device: {device} \nTime already spent: {timer.sum():.3f} sec\nAvg time spent:  {timer.sum()/num_epochs:.3f} sec/epoch  {timer.avg(): .3f} sec/batch')


def train_model_with_Tensorboard(net, train_iter, valid_iter, loss_fn, metrics, optimizer, num_epochs, device):
    pass
