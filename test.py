import os
from presets import ClassificationGreyEval, ClassificationPresetEval
from torch import load as weight_load
from torch import device as pytorch_device
from torch import inference_mode
from torch.backends.cudnn import benchmark, deterministic
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import get_model
from torchvision.transforms.functional import InterpolationMode
import utils

valdir = "/home/local/data/sophie/imagenet/val"
pth_dir = "/home/local/data/sophie/imagenet/output/base"

start_epoch = 0
epochs = 89
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
device = pytorch_device("cuda")




def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg

# create preprocessing transforms
if grey:
    preprocessing = ClassificationGreyEval(
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

for epoch in range(start_epoch, epochs):
    print("Epoch: {}/{} ==================================".format(epoch, epochs))
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
    evaluate(model_without_ddp, criterion, data_loader_test, device=device, print_freq=1563)
