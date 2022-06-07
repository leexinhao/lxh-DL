r"""
Module for training model.
"""

from torch.utils.tensorboard import SummaryWriter
from tools.model_tester import test_model
from tools.utils import Timer, Animator, Accumulator
from tools.metrics import check_metrics
import sys
import torch


def train_batch_L1(net, X, y, loss_fn, metrics, optimizer, device, lamda=0.001):
    r"""
    进行一个batch的带L1正则化训练, 暂时不考虑多GPU训练
    """
    # 暂时不考虑多GPU训练
    X = X.to(device)
    y = y.to(device)
    net.train()

    y_pred = net(X)
    l = loss_fn(y_pred, y)
    regularization_loss = 0
    for param in net.parameters():
        regularization_loss += torch.sum(abs(param))
    l += lamda * regularization_loss
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    train_loss_sum = l.sum()
    train_metrics_sum = []
    for metric in metrics:
        train_metrics_sum.append(metric(y_pred, y, method='sum'))
    return train_loss_sum, train_metrics_sum

def train_batch(net, X, y, loss_fn, metrics, optimizer, device, mutilabel):
    r"""
    进行一个batch的训练, 暂时不考虑多GPU训练
    """
    # 暂时不考虑多GPU训练
    X = X.to(device)
    y = y.to(device)
    net.train()
    y_pred = net(X)
    l = loss_fn(y_pred, y)
    optimizer.zero_grad()
    if mutilabel:
        l.sum().backward()
    else:
        l.backward()
    optimizer.step()
    train_loss_sum = l.sum()
    train_metrics_sum = []
    for metric in metrics:
        train_metrics_sum.append(metric(y_pred, y, method='sum'))
    return train_loss_sum, train_metrics_sum


def train_model(net, train_iter, valid_iter, loss_fn, metrics, optimizer, num_epochs, device, train_func=None, use_animator=False, use_tensorboard=False, print_log=True, log_dir=None, comment='', len_progress=15, multlabel=False):
    r"""
    进行一次完整的训练
    train_func: 每一batch使用的训练函数，若不指定默认为train_batch
    可自定义（比如实现L1正则化），只需要和train_batch的参数及返回值相同即可
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
    if train_func is None:
        train_func = train_batch # 默认训练函数
    # elif type(train_func) is str: # 选择预设默认函数，先不实现，还是在外面定义l1，因为l1有参数
    #     if train_func == 'L1':
    #         train_func = train_batch_with_L1
    #     else:
    #         print(f"未实现{train_func}")
    #         raise NotImplementedError

    _train_model(net, train_iter, valid_iter, loss_fn, metrics, optimizer, num_epochs, device, train_func, animator, writer, print_log, len_progress, multlabel)
    if writer is not None:
        writer.flush()
        writer.close()

def _train_model(net, train_iter, valid_iter, loss_fn, metrics, optimizer, num_epochs, device, train_func, animator, writer, print_log, len_progress, multlabel):
    net = net.to(device)
    print(f"training on {device}")
    num_batches = len(train_iter)
        
    recorder = Accumulator(3 + len(metrics), ["num_instances", "num_elements", "loss", *[f'{metric.name}' for metric in metrics]])
    timer = Timer()
    last_epoch_time = 0
    if writer is not None:
    # 记录初始权值
        for name, param in net.named_parameters():
            writer.add_histogram(name + '_data', param, 0)
    for epoch in range(1, num_epochs+1):
        # 每个epoch开始时重置recorder
        recorder.reset()
        if epoch != 1:
            time_per_epoch = timer.sum() / (epoch - 1)
        else:
            time_per_epoch = -1
        if print_log:
            print(f"Epoch {epoch}/{num_epochs}")
            sys.stdout.write(
                '\r'+f"0/{num_batches} [{len_progress*' '}]")  # 模仿keras输出
        if animator is not None:
            animator.set_suptitle(
                f'Epoch: {epoch}    Device: {device} \nTime already spent: {timer.sum():.4f} sec\nAvg time spent:  {time_per_epoch:.4f} sec/epoch  {timer.avg(): .4f} sec/batch')
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            loss_sum, metrics_sum = train_func(net, features, labels,
                                 loss_fn, metrics, optimizer, device, multlabel)
            recorder.add(len(labels), labels.numel(), loss_sum, *metrics_sum)
            timer.stop()
            if animator is not None:
                animator.set_suptitle(
                    f'Epoch: {epoch}    Device: {device} \nTime already spent: {timer.sum():.4f} sec\nAvg time spent:  {time_per_epoch:.4f} sec/epoch  {timer.avg(): .4f} sec/batch')
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                if animator is not None:
                    animator.add(1, 1, epoch - 1 + (i + 1) / num_batches,
                                (recorder["loss"] / recorder["num_elements"], None))  # loss也先用element算吧
                    for j in range(len(metrics)):
                        animator.add(1, j+2, epoch - 1 + (i + 1) / num_batches,
                                    (recorder[metrics[j].name] / recorder["num_elements"], None)) # metric考虑到有accuracy这样的，还是算单个的
                if writer is not None:
                    writer.add_scalar('Loss/Train', recorder["loss"] / (i + 1), (epoch-1) * num_batches + (i + 1))
                    for j in range(len(metrics)):
                        writer.add_scalar(f'{metrics[j].name}/Train', recorder[metrics[j].name] / recorder["num_exaples"], (epoch-1) * num_batches + (i + 1))
            if print_log:
                tmp_n = min(len_progress, len_progress * (i+1)//num_batches)
                sys.stdout.write(
                        '\r'+f"{i+1}/{num_batches} [{tmp_n*'='+'>'+max(0, len_progress-tmp_n-1)*' '}]")
                        
        if writer is not None:  # 放test_model前面是担心net.eval()对记录造成影响
            # 堆每个epoch，记录梯度，权值
            for name, param in net.named_parameters():
                writer.add_histogram(
                    name + '_grad', param.grad, epoch)
                writer.add_histogram(
                    name + '_data', param, epoch)

        valid_loss, valid_metrics = test_model(net, valid_iter, loss_fn, metrics, device, print_result=False) 
        if animator is not None:
            animator.add(1, 1, epoch, (None, valid_loss))
            for j in range(len(metrics)):
                animator.add(1, j+2, epoch,
                            (None, valid_metrics[j]))
        if writer is not None:
            writer.add_scalar(
                'Loss/Valid', valid_loss, epoch)
            for j in range(len(metrics)):
                writer.add_scalar(f'{metrics[j].name}/Valid', valid_metrics[j], epoch)
        if print_log:
            sys.stdout.write(
                '\r'+f"{num_batches}/{num_batches} [{len_progress*'='}] ")
            print(f"Train Loss: {recorder['loss']/num_batches:.4f}",end=", ")
            for j in range(len(metrics)):
                print(f"Train {metrics[j].name}: {recorder[metrics[j].name] / recorder['num_elements']:.4f}", end=", ")
            print(f"Valid Loss: {valid_loss:.4f}",end=", ")
            for j in range(len(metrics)):
                print(f"Valid {metrics[j].name}: {valid_metrics[j]:.4f}", end=", ")
            print(f"Cost Time {timer.sum()-last_epoch_time:.4f} sec")


        last_epoch_time = timer.sum()
    # 统计一下最终训练效果
    print("Train result")
    print(f'loss {recorder["loss"] / num_batches:.4f}')
    for i, metric in enumerate(metrics):
        print(f'final train {metric.name} {recorder[metric.name] / recorder["num_elements"]:.4f}')
        print(f'final valid {metric.name} {valid_metrics[i]:.4f}')

    print(
        f'{recorder["num_instances"] * num_epochs / timer.sum():.4f} examples/sec on {device}')
    print(f'{timer.sum() / num_epochs:.4f} sec/epoch on {device}')
    if animator is not None:
        animator.set_suptitle(
            f'Epoch: {num_epochs}    Device: {device} \nTime already spent: {timer.sum():.4f} sec\nAvg time spent:  {timer.sum()/num_epochs:.4f} sec/epoch  {timer.avg(): .4f} sec/batch')


