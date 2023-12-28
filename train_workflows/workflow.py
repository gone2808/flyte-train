import os
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import typing
from typing import Optional, List, Tuple
import torch

from tensorboardX import SummaryWriter

from .backbones import get_model
from .train import train, train_one_epoch, set_up_optimizer
from .dataset import DeeplakeDatasetUtils

import flytekit
from flytekit import task, workflow, Resources, ImageSpec, Secret
from flytekit.types.file import PythonPickledFile
from flytekit.types.directory import TensorboardLogs
from flytekitplugins.kfpytorch import PyTorch, Worker

# ai-team
# os.environ["AWS_ACCESS_KEY_ID"] = "qldcAlCPUOoZiQPf0GBQ"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "41NPTd9jOIiWQVaWmAVjUGb9KWpgYmiFhlaIwRTu"
# os.environ["ENDPOINT_URL"] = "http://192.168.1.205:9000"

# local sandbox
os.environ["AWS_ACCESS_KEY_ID"] = "XKUG1tW1aClc0DXwn2Al"
os.environ["AWS_SECRET_ACCESS_KEY"] = "PoAsTNZoSL1V5A8zqDJJzVwu8lkusUcAHms7upeX"
os.environ["ENDPOINT_URL"] = "http://192.168.0.132:9000"



cpu_request = "2000m"
mem_request = "4Gi"
gpu_request = "0"
mem_limit = "8Gi"
gpu_limit = "0"

@dataclass_json
@dataclass
class TrainingConfig(object):
    dataset_name: str = 'COCO2017_val'
    classes_of_interest: List[str] = field(default_factory=lambda: ['person', 'car', 'truck'])
    model_name: str = 'fastercnn_resnet50_fpn'
    pretrain: Optional[str] = None
    batch_size: int = 1
    epochs: int = 1
    log_interval: int = 10
    optimizer: str = 'SGD'
    lr: float = 0.0001
    momentum: float = 0.9
    weight_decay: float = 0.0005


TrainingOutputs = typing.NamedTuple(
    "TrainingOutputs",
    model_state=PythonPickledFile,
    logs=TensorboardLogs,
)

@task(
    task_config=PyTorch(worker=Worker(replicas=2)),
    retries=1,
    cache=False,
    cache_version="0.1",
    requests=Resources(cpu=cpu_request, mem=mem_request, gpu=gpu_request),
    limits=Resources(mem=mem_limit, gpu=gpu_limit),
    environment={"aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
                 "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
                 "endpoint_url": os.environ["ENDPOINT_URL"]},
)
def training_model_task(train_config: TrainingConfig) -> TrainingOutputs:
    # print environment variables
    # print(os.environ["AWS_ACCESS_KEY_ID"])
    # print(os.environ["AWS_SECRET_ACCESS_KEY"])
    # print(os.environ["ENDPOINT_URL"])

    ds_deeplake = DeeplakeDatasetUtils()
    ds_deeplake.set_up_s3_config({"aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
                                  "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
                                  "endpoint_url": os.environ["ENDPOINT_URL"]})
    ds_deeplake.load_dataset_from_s3("s3://datalake/deeplake_datasets/coco2017_val")
    ds_deeplake.set_classes_of_interest(train_config.classes_of_interest)
    print('ds_deeplake INDS_OF_INTEREST:', ds_deeplake.INDS_OF_INTEREST)
    print('ds_deeplake classes_of_interest:', ds_deeplake.classes_of_interest)
    print('ds_deeplake categories', ds_deeplake.categories)

    train_loader = ds_deeplake.to_torch_dataloader(batch_size=train_config.batch_size)
    train_model = get_model('fasterrcnn_resnet50_fpn', num_classes=len(train_config.classes_of_interest))

    # optimizer = set_up_optimizer(train_model)
    params = [p for p in train_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=train_config.lr, momentum=train_config.momentum,
                                weight_decay=train_config.weight_decay)

    log_dir = os.path.join(flytekit.current_context().working_directory, "logs")
    writer = SummaryWriter(log_dir)

    use_cuda = torch.cuda.is_available()
    print(f"Use cuda {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")

    train(train_model, optimizer, train_loader, device, writer, train_config.epochs, train_config.log_interval)
    model_file = os.path.join(
        flytekit.current_context().working_directory,
        f'{train_config.dataset_name}_{train_config.model_name}_{train_config.epochs}'
    )
    torch.save(train_model.state_dict(), model_file)

    return TrainingOutputs(
        model_state=PythonPickledFile(model_file),
        logs=TensorboardLogs(log_dir)
    )


@workflow
def wf(
        train_config: TrainingConfig = TrainingConfig(epochs=5, batch_size=1),
) -> Tuple[PythonPickledFile, TensorboardLogs]:
    # print('aaaa')
    # model = get_training_model(model_name='fasterrcnn_resnet50_fpn')
    # cf = TrainingConfig(epochs = 5, batch_size = 16)
    model, logs = training_model_task(train_config=train_config)
    return model, logs


if __name__ == "__main__":
    wf(train_config=TrainingConfig(
        epochs=5,
        batch_size=1
    ), )
