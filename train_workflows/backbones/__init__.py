import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(name, **kwargs):
    if "pretrained" not in kwargs:
        kwargs["pretrained"] = False
    if "num_classes" not in kwargs:
        kwargs["num_classes"] = 80

    print(kwargs)
    print(kwargs["pretrained"])
    print(kwargs["num_classes"])

    if name == "fasterrcnn_resnet50_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=kwargs["pretrained"])
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 80)
    elif name == "maskrcnn_resnet50_fpn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=kwargs["pretrained"])
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, kwargs["num_classes"])
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # Get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 80)


    return model