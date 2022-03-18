r"""
Module for testing model.
"""

import torch
from tools.utils import Accumulator, Timer
from tools.metrics import check_metrics
def test_model(net, test_iter, loss_fn, metrics, device):
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
            recorder.add(loss_fn(y_pred, y).item(),
            *[metric(y_pred, y, method="sum") for metric in metrics])
            timer.stop()
    test_loss = recorder['loss'] / num_batches
    test_metrics = [recorder[metric.name] / num_instances for metric in metrics]
    
    print(
        f"""        Test result
         Number of instances: {num_instances}
         Number of batches: {num_batches}
         Avg loss of each batch: {test_loss:>8f}
         """)
    for i, test_metric in enumerate(test_metrics):
        print(f"{metrics[i].name}: {test_metric:.2f}")
    print(f"""Time spent: {timer.sum()} sec
            {num_instances / timer.sum():.3f} examples/sec
            {timer.avg():.3f} sec/batch on {device}
            """)
    return test_loss, test_metrics

