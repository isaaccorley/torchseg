from typing import Callable, Optional

import torch.nn as nn

from ...base import ClassificationHead, SegmentationHead, SegmentationModel
from ...encoders import get_encoder
from .decoder import UnetPlusPlusDecoder


class UnetPlusPlus(SegmentationModel):
    """
    Unet++ is a fully convolution neural network for image semantic segmentation.
    Consist of *encoder* and *decoder* parts connected with *skip connections*.
    Encoder extract features of different spatial resolution (skip connections) which
    are used by decoder to define accurate segmentation mask. Decoder of Unet++ is
    more complex than in usual Unet.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage
            generate features two times smaller in spatial dimensions than previous one
            (e.g. for depth 0 we will have features with shapes [(N, C, H, W),], for
            depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on). Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"**
            (pre-training on ImageNet) and other pretrained weights (see table with
            available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for
            convolutions used in decoder. Length of the list should be the
            same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D
            and Activation layers is used. Available options are **True, False**
        decoder_attention_type: Attention module used in decoder of the model.
            Options are **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of
            channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**,
            **"tanh"**, **"identity"**, **callable** and **None**. Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output
            (classification head). Auxiliary output is build on top of encoder if
            **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        head_upsampling: Factor to upsample input to segmentation head. Defaults to 1.
            This allows for use of U-Net decoder with models that need additional
            upsampling to be at the original input image resolution.

    Reference:
        https://arxiv.org/abs/1807.10165

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_indices: Optional[tuple[int]] = None,
        encoder_depth: int = 5,
        encoder_output_stride: int = 32,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Callable = nn.Identity(),
        encoder_params: dict = {},
        aux_params: Optional[dict] = None,
        head_upsampling: int = 1,
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError(f"UnetPlusPlus is not support encoder_name={encoder_name}")

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            indices=encoder_indices,
            depth=encoder_depth,
            output_stride=encoder_output_stride,
            weights=encoder_weights,
            **encoder_params,
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=head_upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = f"unetplusplus-{encoder_name}"
        self.initialize()
