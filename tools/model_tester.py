r"""
Module for testing model.
"""

import torch
from tools.utils import Accumulator, Timer
from tools.metrics import check_metrics


def test_model(net, test_iter, loss_fn, metrics, device, print_result=True):
    # check metrics
    metrics = check_metrics(metrics)
    num_instances = len(test_iter.dataset)
    num_batches = len(test_iter)
    net.eval()
    recorder = Accumulator(1+len(metrics), ['loss', *[metric.name for metric in metrics]])
    timer = Timer()
    with torch.no_grad():
        for X, y in test_iter:
            timer.start()
            X, y = X.to(device), y.to(device)
            y_pred = net(X)
            recorder.add(loss_fn(y_pred, y).sum().item(),
            *[metric(y_pred, y, method="sum") for metric in metrics])
            timer.stop()
    size_element = y.numel() / len(y) # 默认送进来的是数值不是独热码，故分类任务size_element=1
    num_elements = num_instances * size_element  
    test_loss = recorder['loss'] / num_instances
    test_metrics = [recorder[metric.name] / num_elements for metric in metrics]
    
    if print_result:
        print("Test result")
        print(f"Number of instances: {num_instances}")
        print(f"Number of batches: {num_batches}")
        print(f"Size of an element: {num_elements}")
        print(f"Avg loss of each instance: {test_loss:>8f}")
        for i, test_metric in enumerate(test_metrics):
            print(f"{metrics[i].name}: {test_metric:.4f}")
        print(f"Time spent: {timer.sum()} sec")
        print(f"{num_instances / timer.sum():.4f} examples/sec")
        print(f"{timer.avg():.4f} sec/batch on {device}")
                
    return test_loss, test_metrics

