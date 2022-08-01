import math
import torch
import torchvision
from ..external import pointnet_utils
from ..utils import modules


# TODO: Encoder mixin
class PointNet(torch.nn.Module):
    ORIGINAL_LATENT_DIMS = 1024

    def __init__(self, latent_dims, pretrained, has_input_normals):
        super().__init__()
        assert isinstance(latent_dims, int) and latent_dims > 0
        assert isinstance(pretrained, bool)

        # instantiate layers
        channel = {
            False: 3,   # 3D point
            True: 6     # 3D point + 3D surface normal
        }[has_input_normals]
        self.core = pointnet_utils.PointNetEncoder(
            global_feat=True, feature_transform=False, channel=channel
        )
        if pretrained:
            # TODO: pretrain with the implementation given by 
            #       https://github.com/yanx27/Pointnet_Pointnet2_pytorch
            raise NotImplementedError

        # Inspired from https://github.com/bednarikjan/differential_surface_representation/blob/master/encoder.py#L129
        self.output = modules.build_linear_relu(
            self.ORIGINAL_LATENT_DIMS, latent_dims
        )

    def forward(self, pcl_nml):
        """
        Args:
            pcl_nml (torch.Tensor): Batch of 3D point clouds, and optionally
                                    surface normals, with shape (N, P, 3/6)
        """
        # `pointnet_utils.PointNetEncoder` expects inputs of shape (N, C, P)
        pcl_nml = pcl_nml.transpose(1, 2)       # (N, 3/6, P)
        latent_code, _, _ = self.core(pcl_nml)  # (N, ORIGINAL_LATENT_DIMS)
        return self.output(latent_code)         # (N, latent_code_dims)


class ResNet18(torch.nn.Module):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, latent_dims, pretrained):
        super().__init__()
        assert isinstance(latent_dims, int) and latent_dims > 0
        assert isinstance(pretrained, bool)

        self.pretrained = pretrained
        if pretrained:
            # Reference: https://pytorch.org/vision/stable/models.html#codecell2
            self.normalize = torchvision.transforms.Normalize(
                mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD
            )
        self.core = torchvision.models.resnet18(pretrained=pretrained)
        # replace classifier fully connected layer
        self.core.fc = modules.build_linear_relu(
            self.core.fc.in_features, latent_dims
        )

    def forward(self, image):
        """
        Args:
            pcl_nml (torch.Tensor): Batch of images with shape (N, C, H, W)
        """
        if self.pretrained:
            image = self.normalize(image)
        return self.core(image)


class PositionalEncoder(torch.nn.Module):
    """
    Implementation references:
        1. https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L22
        2. https://github.com/lioryariv/idr/blob/main/code/model/embedder.py
        3. https://github.com/bmild/nerf/issues/12
    """

    def __init__(
        self,
        num_pos_encoding_octaves,
        euclidean_space_scale,
        include_input                       # [Reference 3]
    ):
        super().__init__()
        assert isinstance(num_pos_encoding_octaves, int) \
               and num_pos_encoding_octaves > 0
        assert isinstance(euclidean_space_scale, (int, float)) \
               and euclidean_space_scale > 0.
        assert isinstance(include_input, bool)
        self.include_input = include_input

        # deduce the dimensions of a positional-encoded scalar
        self.dims = 2 * num_pos_encoding_octaves
        if include_input:
            self.dims += 1

        # collate the positional encoding angular frequencies
        scale = 1 / euclidean_space_scale   # [Reference 3]
        self.register_buffer(
            "omega",
            2 ** torch.arange(num_pos_encoding_octaves) * math.pi * scale
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input data of arbitrary shape
        Returns:
            pos_encoded_input (torch.Tensor): Positional-encoded input data
                                              with shape
                                              (*input.shape, self.dims)
        """
        input_shape_len = len(input.shape)
        reshaped_omega = self.omega.view(               # (input_shape_len of 1s, num_pos_encoding_octaves)
            *([1] * input_shape_len), -1
        )
        reshaped_input = input.unsqueeze(-1)            # (*input.shape, 1)
        scaled_input = reshaped_omega * reshaped_input  # (*input.shape, num_pos_encoding_octaves)

        pos_encoded_input = torch.sin(scaled_input)     # (*input.shape, num_pos_encoding_octaves)
        pos_encoded_input = torch.cat(                  # (*input.shape, 2 * num_pos_encoding_octaves)
            (pos_encoded_input, torch.cos(scaled_input)), dim=-1
        )
        if self.include_input:
            pos_encoded_input = torch.cat(              # (*input.shape, 2 * num_pos_encoding_octaves)
            (reshaped_input, pos_encoded_input), dim=-1 # (*input.shape, 2 * num_pos_encoding_octaves + 1 / self.dims)
        )

        return pos_encoded_input
