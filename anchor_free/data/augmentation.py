import albumentations as A
from albumentations.augmentations import CoarseDropout


augmentations = A.Compose(
    [
        A.OneOf(
            [
                # A.Affine(
                #    translate_percent=(-0.0625, 0.0625),
                #    scale=(-0.1, 0.1),
                #    rotate=(-45, 45),
                #    shear=(-15, 15),
                # ),
                A.ShiftScaleRotate(),
                # A.RandomResizedCrop(256, 256),
                A.HorizontalFlip(),
                # A.VerticalFlip(),
                # A.Transpose(),
            ],
            p=1.0,
        ),
        A.OneOf(
            [
                A.RandomGamma(),
                A.RGBShift(),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1),
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20,
                ),
            ],
            p=1.0,
        ),
        A.OneOf(
            [
                A.ElasticTransform(
                    alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                ),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ],
            p=0.0,
        ),
        A.OneOf(
            [A.GaussNoise(p=0.5), A.Blur(p=0.5), CoarseDropout(max_holes=5)], p=0.0
        ),
    ],
    p=1,
    bbox_params=A.BboxParams(format="pascal_voc", min_area=16, min_visibility=0.1),
)
