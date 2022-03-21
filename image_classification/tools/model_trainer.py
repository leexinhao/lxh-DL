r"""
Module for training model.
"""

from torch.utils.tensorboard import SummaryWriter
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


def train_model(net, train_iter, valid_iter, loss_fn, metrics, optimizer, num_epochs, device, use_animator=False, use_tensorboard=False, log_dir=None, comment=''):
    r"""
    进行一次完整的训练 
    """
    # check metrics
    metrics = check_metrics(metrics)
    if use_tensorboard:
        writer = SummaryWriter(log_dir=log_dir, comment=comment)
    else:
        writer = None
    if use_animator:
        n_columns = 1 + len(metrics)
        animator = Animator(legends=[['Loss/Train', 'Loss/Valid'],
                                    *[[f'{metric.name}/Train', f'{metric.name}/Valid'] for metric in metrics]],
                            nrows=1, ncols=n_columns, figsize=(8*n_columns, 8),
                            xlabels=['epoch' for _ in range(n_columns)],
                            ylabels=['loss', *[metric.name for metric in metrics]],
                            xlims=[[0, num_epochs] for _ in range(n_columns)],
                            ylims=None)
    else:
        animator = None
    _train_model(net, train_iter, valid_iter, loss_fn, metrics, optimizer, num_epochs, device, animator, writer)
    if writer is not None:
        writer.flush()
        writer.close()

def _train_model(net, train_iter, valid_iter, loss_fn, metrics, optimizer, num_epochs, device, animator, writer):
    net = net.to(device)
    print(f"training on {device}")
    num_batches = len(train_iter)
        
    recorder = Accumulator(2 + len(metrics), ["num_exaples", "loss", *[f'{metric.name}' for metric in metrics]])
    timer = Timer()
    if writer is not None:
    # 记录初始权值
        for name, param in net.named_parameters():
            writer.add_histogram(name + '_data', param, 0)
    for epoch in range(num_epochs):
        # 每个epoch开始时重置recorder
        recorder.reset()
        if epoch != 0:
            time_per_epoch = timer.sum() / epoch
        else:
            time_per_epoch = -1
        if animator is not None:
            animator.set_suptitle(
                f'Epoch: {epoch+1}    Device: {device} \nTime already spent: {timer.sum():.3f} sec\nAvg time spent:  {time_per_epoch:.3f} sec/epoch  {timer.avg(): .3f} sec/batch')
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            loss_sum, metrics_sum = train_batch(net, features, labels,
                                 loss_fn, metrics, optimizer, device)
            recorder.add(labels.shape[0], loss_sum, *metrics_sum)
            timer.stop()
            if animator is not None:
                animator.set_suptitle(
                    f'Epoch: {epoch+1}    Device: {device} \nTime already spent: {timer.sum():.3f} sec\nAvg time spent:  {time_per_epoch:.3f} sec/epoch  {timer.avg(): .3f} sec/batch')
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                if animator is not None:
                    animator.add(1, 1, epoch + (i + 1) / num_batches,
                                (recorder["loss"] / (i + 1), None))
                    for j in range(len(metrics)):
                        animator.add(1, j+2, epoch + (i + 1) / num_batches,
                                    (recorder[metrics[j].name] / recorder["num_exaples"], None)) 
                if writer is not None:
                    writer.add_scalar('Loss/Train', recorder["loss"] / (i + 1), epoch * num_batches + (i + 1))
                    for j in range(len(metrics)):
                        writer.add_scalar(f'{metrics[j].name}/Train', recorder[metrics[j].name] / recorder["num_exaples"], epoch * num_batches + (i + 1))
        if writer is not None:  # 放test_model前面是担心net.eval()对记录造成影响
            # 堆每个epoch，记录梯度，权值
            for name, param in net.named_parameters():
                writer.add_histogram(
                    name + '_grad', param.grad, epoch+1)
                writer.add_histogram(
                    name + '_data', param, epoch+1)

        valid_loss, valid_metrics = test_model(net, valid_iter, loss_fn, metrics, device, print_result=False) 
        if animator is not None:
            animator.add(1, 1, epoch + 1, (None, valid_loss))
            for j in range(len(metrics)):
                animator.add(1, j+2, epoch + 1,
                            (None, valid_metrics[j]))
        if writer is not None:
            writer.add_scalar(
                'Loss/Valid', valid_loss, epoch + 1)
            for j in range(len(metrics)):
                writer.add_scalar(f'{metrics[j].name}/Valid', valid_metrics[j], epoch + 1)

    # 统计一下最终训练效果
    print("Train result")
    print(f'loss {recorder["loss"] / num_batches:.3f}')
    for i, metric in enumerate(metrics):
        print(f'final train {metric.name} {recorder[metric.name] / recorder["num_exaples"]:.3f}')
        print(f'final valid {metric.name} {valid_metrics[i]:.3f}')

    print(
        f'{recorder["num_exaples"] * num_epochs / timer.sum():.3f} examples/sec on {device}')
    print(f'{timer.sum() / num_epochs:.3f} sec/epoch on {device}')
    if animator is not None:
        animator.set_suptitle(
            f'Epoch: {num_epochs}    Device: {device} \nTime already spent: {timer.sum():.3f} sec\nAvg time spent:  {timer.sum()/num_epochs:.3f} sec/epoch  {timer.avg(): .3f} sec/batch')


