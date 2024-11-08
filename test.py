import csv
import os
from presets import ClassificationGreyEval, ClassificationPresetEval, ClassificationSingleChannelGreyEval
from sklearn.metrics import roc_auc_score
from torch import load as weight_load
from torch import device as pytorch_device
from torch import arange, cat, inference_mode, round, tensor
from torch.backends.cudnn import benchmark, deterministic
from torch.nn import Conv2d, CrossEntropyLoss
from torch.nn.functional import softmax
from torch.utils.data import SequentialSampler, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import get_model
from torchvision.transforms.functional import InterpolationMode
import utils

from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy, MulticlassAUPRC, MulticlassF1Score

test_model = "base" #"single" #"grey"
valdir = "/home/local/data/sophie/imagenet/val"
# pth_dir = "/home/local/data/sophie/imagenet/output/{}".format(test_model)
start_epoch = 130
epochs = 138
pth_dir = "/home/local/data/sophie/imagenet/output/{}/continued".format(test_model)
# start_epoch = 120
# epochs = 122

output_csv = "/home/local/data/sophie/imagenet/output/{}_results.csv".format(test_model)
val_crop_size = 224
val_resize_size = 256
batch_size = 32
workers = 16
interpolation = InterpolationMode("bilinear")
backend = "PIL"
usev2 = False
architecture = "resnet50"
weights = None
grey = True if "grey" in pth_dir else False
single = True if "single" in pth_dir else False
device = pytorch_device("cuda")

def convert_to_single_channel(model):
    """
    Modifies the first convolutional layer of a given model to accept single-channel input.

    Args:
        model (torch.nn.Module): The model to be modified.

    Returns:
        torch.nn.Module: The modified model with a single-channel input.
    """
    # Identify the first convolutional layer
    conv1 = None
    for name, layer in model.named_modules():
        if isinstance(layer, Conv2d):
            conv1 = layer
            conv1_name = name
            break

    if conv1 is None:
        raise ValueError("The model does not have a Conv2D layer.")

    # Create a new convolutional layer with the same parameters except for the input channels
    new_conv1 = Conv2d(
        in_channels=1,  # Change input channels to 1
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None
    )

    # Replace the old conv1 layer with the new one
    def recursive_setattr(model, attr, value):
        attr_list = attr.split('.')
        for attr_name in attr_list[:-1]:
            model = getattr(model, attr_name)
        setattr(model, attr_list[-1], value)

    recursive_setattr(model, conv1_name, new_conv1)

    return model

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"
    roc_output = tensor([])
    roc_target = tensor([])
    num_processed_samples = 0
    auroc = MulticlassAUROC(num_classes=1000,device=device)
    f1 = MulticlassF1Score(num_classes=1000,device=device)
    auprc = MulticlassAUPRC(num_classes=1000,device=device)
    multiacc = MulticlassAccuracy(num_classes=1000, device=device)
    with inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # roc_auc = roc_auc_score(target.cpu(), \
            #     torch.nn.functional.softmax(output.cpu(), dim=1) / torch.nn.functional.softmax(output.cpu(), dim=1).sum(dim=1, keepdim=True), \
            #     multi_class='ovo', labels=torch.arange(0,1000))
            # roc_output = cat((roc_output, output.cpu()),0)
            # roc_target = cat((roc_target, target.cpu()),0)
            # auroc.update(output, target)
            # f1.update(output, target)
            auprc.update(output, target)
            # multiacc.update(output, target)
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            # print("auroc: {} ".format( auroc.compute()))
            # metric_logger.meters["auroc"].update(roc_auc.item(), n=batch_size)
            num_processed_samples += batch_size
    # roc_auc = roc_auc_score(roc_target, softmax(roc_output, dim=1) / softmax(roc_output, dim=1).sum(dim=1, keepdim=True), multi_class='ovo', labels=arange(0,1000))
    # avg_auroc = auroc.compute().item()
    # avg_f1 = f1.compute().item()
    avg_auprc = auprc.compute().item()
    # avg_multiacc = multiacc.compute().item()
    # metric_logger.meters["auroc"].update(roc_auc.item(), n=num_processed_samples)
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} AUPRC {avg_auprc} ")#AUROC {metric_logger.auroc.global_avg:.3f}")
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, avg_auprc#, metric_logger.auroc.global_avg

# create preprocessing transforms
if grey:
    preprocessing = ClassificationGreyEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        interpolation=interpolation,
        backend=backend,
        use_v2=usev2,
    )
elif single:
    preprocessing = ClassificationSingleChannelGreyEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        interpolation=interpolation,
        backend=backend,
        use_v2=usev2,
    )
else:
    preprocessing = ClassificationPresetEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        interpolation=interpolation,
        backend=backend,
        use_v2=usev2,
    )
# val data folder
dataset_test = ImageFolder(
    valdir,
    preprocessing,
)
# create data loader
test_sampler = SequentialSampler(dataset_test)

num_classes = len(dataset_test.classes)

# val loader
data_loader_test = DataLoader(
    dataset_test, batch_size=batch_size,
    sampler=test_sampler, num_workers=workers,
    pin_memory=True
)

# define test criterion
criterion = CrossEntropyLoss(label_smoothing=0.0)

model = get_model(architecture, weights=weights,
    num_classes=num_classes)
if single:
    model = convert_to_single_channel(model)
fieldnames = ['Epoch', 'AP', 'Top1', 'Top5']

# with open(output_csv, 'w', newline='') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()

for epoch in range(start_epoch, epochs):
    print("Epoch: {}/{} ==================================".format(epoch, epochs))
    # print("Epoch: {}/{} ==================================".format(start_epoch, epochs))
    # epoch = start_epoch
    model_without_ddp = model
    # load epoch weights
    weights = weight_load(os.path.join(pth_dir,"model_{}.pth".format(epoch)),
        map_location="cpu", weights_only=False)
    model_without_ddp.load_state_dict(weights["model"])
    # move to GPU
    model_without_ddp.to(device)
    # ensure results are consistent by disabling benchmarking
    benchmark = False
    deterministic = True
    acc1,acc5,auprc = evaluate(model_without_ddp, criterion, data_loader_test, device=device, print_freq=1563)
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerows([
            {'Epoch':epoch+1, 'AP':auprc, 'Top1':acc1, 'Top5':acc5},
        ])
