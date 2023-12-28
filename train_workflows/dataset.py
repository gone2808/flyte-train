import torch
import numpy as np
import deeplake
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class DeeplakeDatasetUtils(object):
    def __init__(self):
        self.deeplake_ds = None
        self.__s3_config = None
        self.transforms = None
        self.categories = None
        self.classes_of_interest = None
        self.INDS_OF_INTEREST = None
    
    def set_up_s3_config(self, s3_config: dict):
        self.__s3_config = s3_config

    def set_up_transforms(self, transforms):
        self.transforms = transforms

    def set_deeplake_ds(self, deeplake_ds: deeplake.core.dataset.dataset.Dataset):
        self.deeplake_ds = deeplake_ds
    
    def load_dataset_from_s3(self, dataset_path: str):
        ds = deeplake.load(path=dataset_path, creds=self.__s3_config)
        print(type(ds))
        self.deeplake_ds = ds
        # add creds key for ds to be able to load images from image path(deeplake)
        self.__add_creds_dataset()
        self.categories = self.deeplake_ds.tensors['categories'].info['class_names']
        self.classes_of_interest = self.categories
        self.INDS_OF_INTEREST = [ds.categories.info.class_names.index(item) for item in self.classes_of_interest]
        # print(len([ds.categories.info.class_names.index(item) for item in self.classes_of_interest]))

    def __add_creds_dataset(self, creds_name='my_s3_creds'):
        if creds_name not in self.deeplake_ds.get_creds_keys():
            self.deeplake_ds.add_creds_key(creds_name, managed=False)
            # print(f"creds {creds_name} already exists")
            # return
        self.deeplake_ds.populate_creds(creds_name, self.__s3_config)
        print(f"creds {creds_name} added")

    def get_all_deeplake_ds_classes(self):
        return self.deeplake_ds.tensors['categories'].info['class_names']

    def set_classes_of_interest(self, classes_of_interest: list):
        # check if interested classes are in the categories
        for c in classes_of_interest:
            if c not in self.get_all_deeplake_ds_classes():
                raise ValueError(f"{c} is not in the categories")
        self.classes_of_interest = classes_of_interest
        self.INDS_OF_INTEREST = [self.deeplake_ds.categories.info.class_names.index(item) for item in self.classes_of_interest]

    def transform_train(self, sample_in):

        # Convert any grayscale images to RGB
        image = sample_in['images']
        shape = image.shape
        if shape[2] == 1:
            image = np.repeat(image, int(3 / shape[2]), axis=2)

        # Convert boxes to Pascal VOC format
        boxes = coco_2_pascal(sample_in['boxes'], shape)

        # Filter only the labels that we care about for this training run
        labels_all = sample_in['categories']
        indices = [l for l, label in enumerate(labels_all) if label in self.INDS_OF_INTEREST]
        labels_filtered = labels_all[indices]
        labels_remapped = [self.INDS_OF_INTEREST.index(label) for label in labels_filtered]
        boxes_filtered = boxes[indices, :]

        # Make sure the number of labels and boxes is still the same after filtering
        assert (len(labels_remapped)) == boxes_filtered.shape[0]

        WIDTH = 128
        HEIGHT = 128
        tform_train = A.Compose([
            A.RandomSizedBBoxSafeCrop(width=WIDTH, height=HEIGHT, erosion_rate=0.2),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'bbox_ids'], min_area=16,
                                    min_visibility=0.6))


        # Pass all data to the Albumentations transformation
        transformed = tform_train(image=image,
                                  bboxes=boxes_filtered,
                                  bbox_ids=np.arange(boxes_filtered.shape[0]),
                                  class_labels=labels_remapped,
                                  )

        # Convert boxes and labels from lists to torch tensors, because Albumentations does not do that automatically.
        # Be very careful with rounding and casting to integers, becuase that can create bounding boxes with invalid dimensions
        labels_torch = torch.tensor(transformed['class_labels'], dtype=torch.int64)

        boxes_torch = torch.zeros((len(transformed['bboxes']), 4), dtype=torch.int64)
        for b, box in enumerate(transformed['bboxes']):
            boxes_torch[b, :] = torch.tensor(np.round(box))

        # Put annotations in a separate object
        target = {'labels': labels_torch, 'boxes': boxes_torch}

        return transformed['image'], target

    def transform_val(self, sample_in):

        # Convert any grayscale images to RGB
        image = sample_in['images']
        shape = image.shape
        if shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        # Convert boxes to Pascal VOC format
        boxes = coco_2_pascal(sample_in['boxes'], shape)


        WIDTH = 128
        HEIGHT = 128
        tform_val = A.Compose([
            A.Resize(width=WIDTH, height=HEIGHT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'bbox_ids'], min_area=8,
                                    min_visibility=0.6))
        # Pass all data to the Albumentations transformation
        transformed = tform_val(image=image,
                                bboxes=boxes,
                                bbox_ids=np.arange(boxes.shape[0]),
                                class_labels=sample_in['labels'],
                                )

        # Convert boxes and labels from lists to torch tensors, because Albumentations does not do that automatically.
        # Be very careful with rounding and casting to integers, becuase that can create bounding boxes with invalid dimensions
        labels_torch = torch.tensor(transformed['class_labels'], dtype=torch.int64)

        boxes_torch = torch.zeros((len(transformed['bboxes']), 4), dtype=torch.int64)
        for b, box in enumerate(transformed['bboxes']):
            boxes_torch[b, :] = torch.tensor(np.round(box))

        # Put annotations in a separate object
        target = {'labels': labels_torch, 'boxes': boxes_torch}

        # We also return the shape of the original image in order to resize the predictions to the dataset image size
        return transformed['image'], target, sample_in['index'], shape

    def to_torch_dataloader(self, batch_size=1):
        # self.deeplake_ds.save_view(id="all")
        self.deeplake_ds[:10].save_view(id="all")
        self.deeplake_ds.get_view("all").optimize()
        ds_view = self.deeplake_ds.load_view("all")
        train_loader = ds_view.pytorch(num_workers=2, shuffle=False, transform=self.transform_train,
                                       tensors=['images', 'categories', 'boxes'],
                                       batch_size=batch_size,
                                       collate_fn=collate_fn)
        # train_loader = deeplake.integrations.pytorch.dataset_to_pytorch(dataset=self.deeplake_ds,
        #                                                                 num_workers=4,
        #                                                                 batch_size=1,
        #                                                                 drop_last=False,
        #                                                                 collate_fn=collate_fn,
        #                                                                 pin_memory=False,
        #                                                                 shuffle=True,
        #                                                                 buffer_size=4096,
        #                                                                 use_local_cache=False,
        #                                                                 transform=self.transform_train,
        #                                                                 tensors=['images', 'categories', 'boxes'],
        #                                                                 )

        # print(type(train_loader))
        return train_loader


# Conversion script for bounding boxes from coco to Pascal VOC format
def coco_2_pascal(boxes, shape):
    # Convert bounding boxes to Pascal VOC format and clip bounding boxes to make sure they have non-negative width and height

    return np.stack((np.clip(boxes[:, 0], 0, None), np.clip(boxes[:, 1], 0, None),
                     np.clip(boxes[:, 0] + np.clip(boxes[:, 2], 1, None), 0, shape[1]),
                     np.clip(boxes[:, 1] + np.clip(boxes[:, 3], 1, None), 0, shape[0])), axis=1)


def model_2_image(boxes, model_shape, img_shape):
    # Resize the bounding boxes convert them from Pascal VOC to COCO

    m_h, m_w = model_shape
    i_h, i_w = img_shape

    x0 = boxes[:, 0] * (i_w / m_w)
    y0 = boxes[:, 1] * (i_h / m_h)
    x1 = boxes[:, 2] * (i_w / m_w)
    y1 = boxes[:, 3] * (i_h / m_h)

    return np.stack((x0, y0, x1 - x0, y1 - y0), axis=1)


def collate_fn(batch):
    return tuple(zip(*batch))

