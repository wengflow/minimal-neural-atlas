import abc
import numpy as np
import torch
import pytorch3d.transforms


class Transform(abc.ABC):   # Mixin
    """
    NOTE:
        1. Tensors that are instantiated in this class and are also involved in 
           operations with the input data should have the same data-type 
           (whenever possible) and reside in the same device as the input
        2. To transfer the complexity of data-type and device conversions to
           `_before_apply()`, we assume that input to `_before_apply()` and 
           `apply_same()` have the same data-type and reside in the same device
        3. Operations on the input data should be out-of-place to enable 
           gradient computation
    """
    def apply(self, *args, **kwargs):
        self._before_apply(*args, **kwargs)
        return self.apply_same(*args, **kwargs)

    @abc.abstractmethod
    def _before_apply(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def apply_same(self, *args, **kwargs):
        """
        Precondition: `self.apply()` has been called at least once
        """
        pass


class Trim(Transform):
    def __init__(self, trim_vector_dim, trim_negative):
        super().__init__()
        assert isinstance(trim_vector_dim, int)
        assert isinstance(trim_negative, bool)
        self.trim_vector_dim = trim_vector_dim
        self.trim_negative = trim_negative

    def _before_apply(self, input):
        """
        Args:
            input (torch.Tensor): Input data with the size of the last 
                                  dimension at least `self.trim_vector_dim + 1`
        """
        self.is_negative = input[..., self.trim_vector_dim] < 0.

    def apply_same(self, input):
        """
        Args:
            input (torch.Tensor): Input data with the size of the last 
                                  dimension at least `self.trim_vector_dim + 1`
        NOTE:
            If `torch.sum(self.is_negative) == 1`, the output will be of shape
            (input.shape[-1], ) instead of
            (torch.sum(self.is_negative), input.shape[-1]) in the general case
        """
        if self.trim_negative:
            return input[~self.is_negative, :]
        else:
            return input[self.is_negative, :]


class RandomSubsampling(Transform):
    def __init__(self, len, axis, replacement):
        super().__init__()
        assert isinstance(len, int) and len >= 0
        assert isinstance(replacement, bool)
        assert isinstance(axis, int) and axis >= 0
        self.len = len
        self.replacement = replacement
        self.axis = axis

    def _before_apply(self, input):
        """
        Args:
            input (torch.Tensor): Input data of arbitrary shape
        """
        input_len = input.shape[self.axis]
        if self.replacement:
            self.sampled_indices = np.random.randint(input_len, size=self.len)
        else:
            self.sampled_indices = np.random.choice(
                input_len, size=self.len, replace=False
            )

    def apply_same(self, input):
        """
        Args:
            input (torch.Tensor): Input data of arbitrary shape
        """
        num_input_axes = len(input.shape)
        slicing_indices = num_input_axes * [ slice(None) ]
        slicing_indices[self.axis] = self.sampled_indices
        return input[slicing_indices]


class SequentialSubsampling(Transform):
    def __init__(self, len, axis):
        super().__init__()
        assert isinstance(len, int) and len >= 0
        assert isinstance(axis, int) and axis >= 0
        self.len = len
        self.axis = axis

    def _before_apply(self, input):
        """
        Args:
            input (torch.Tensor): Input data of arbitrary shape
        """
        pass

    def apply_same(self, input):
        """
        Args:
            input (torch.Tensor): Input data of arbitrary shape
        """
        num_input_axes = len(input.shape)
        slicing_indices = num_input_axes * [ slice(None) ]
        slicing_indices[self.axis] = slice(self.len)
        return input[slicing_indices]


class Random3DRotation(Transform):
    def __init__(self):
        super().__init__()

    def _before_apply(self, pcl_dirvec):
        if pcl_dirvec.is_floating_point():
            mat_dtype = pcl_dirvec.dtype
        else:
            mat_dtype = torch.promote_types(
                torch.get_default_dtype(), pcl_dirvec.dtype
            )        
        self.random_rotation_mat = pytorch3d.transforms.random_rotation(
            dtype=mat_dtype, device=pcl_dirvec.device
        )

    def apply_same(self, pcl_dirvec):
        """
        Args:
            pcl_dirvec (torch.Tensor): 3D point cloud or direction vector
                                       tensor with shape (..., 3)
        """
        return pcl_dirvec @ self.random_rotation_mat


class RandomReflection(Transform):
    def __init__(self, reflection_vector_dim, input_vector_dims, p=0.5):
        super().__init__()
        assert isinstance(reflection_vector_dim, int)
        assert isinstance(input_vector_dims, int)
        assert isinstance(p, float)
        self.reflection_vector = torch.ones(input_vector_dims)
        self.reflection_vector[reflection_vector_dim] = -1
        self.p = p

    def _before_apply(self, input):        
        self.reflect = torch.bernoulli(input=torch.tensor(self.p)) \
                            .bool().item()
        self.reflection_vector = self.reflection_vector.to(
            device=input.device, dtype=input.dtype
        )

    def apply_same(self, input):
        """
        Args:
            input (torch.Tensor): Input data with shape (..., input_dims)
        """
        if not self.reflect:
            return input
        
        # Out-of-place equivalent of
        # `input[reflection_vector_dim] = -input[reflection_vector_dim]
        #  return input`
        return input * self.reflection_vector


class GaussianNoise(Transform):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def _before_apply(self, input):
        """
        Args:
            input (torch.Tensor): Input data of arbitrary shape
        """
        if input.is_floating_point():
            noise_dtype = input.dtype
        else:
            noise_dtype = torch.promote_types(
                torch.get_default_dtype(), input.dtype
            )
        self.noise = self.mean + self.std * torch.randn(
            input.shape, dtype=noise_dtype, device=input.device
        )

    def apply_same(self, input):
        """
        Args:
            input (torch.Tensor): Input data with equal number of axes /
                                  dimensions as `self.noise`
        """
        if input.shape == self.noise.shape:
            return input + self.noise
        
        input_shape_tensor = torch.as_tensor(input.shape)
        noise_shape_tensor = torch.as_tensor(self.noise.shape)
        if not torch.all(input_shape_tensor <= noise_shape_tensor):
            # extend `self.noise` and preserve the existing `self.noise` values
            new_noise_shape = torch.maximum(
                input_shape_tensor, noise_shape_tensor
            )
            new_noise = self.mean + self.std * torch.randn(
                *new_noise_shape, dtype=self.noise.dtype,
                device=self.noise.device
            )
            slicing_indices = list(map(slice, noise_shape_tensor))
            new_noise[slicing_indices] = self.noise
            self.noise = new_noise

        slicing_indices = list(map(slice, input_shape_tensor))
        sliced_noise = self.noise[slicing_indices]

        return input + sliced_noise


class PclNormalization(Transform):
    def __init__(self, mode, scale):
        """Normalize point clouds such that the bounding ball or
        cube has its center at the origin and has a particular scale.

        Args:
            mode ({ "ball", "cube" }): Point cloud normalization mode
            scale (float or int): Radius of bounding ball or half side length
                                  of the bounding cube of the normalized 
                                  point cloud
        """
        super().__init__()
        assert mode in [ "ball", "cube" ]
        assert isinstance(scale, (int, float)) and scale > 0.
        self.mode = mode
        self.target_scale = scale

    def _before_apply(self, pcl):
        """
        Args:
            pcl (torch.Tensor): Tensor of K-D point clouds of size P with 
                                shape (..., P, K)
        """        
        if self.mode == "ball":
            if not pcl.is_floating_point():
                pcl = pcl.to(torch.promote_types(
                    torch.get_default_dtype(), pcl.dtype
                ))

            self.translation = pcl.mean(dim=-2, keepdim=True)   # (..., 1, K)
            centered_pcl = pcl - self.translation               # (..., P, K)
            unit_scale = torch.linalg.vector_norm(              # (..., 1, 1)
                centered_pcl, dim=-1, keepdim=True
            ).max(dim=-2, keepdim=True).values
            self.scale = unit_scale / self.target_scale         # (..., 1, 1)

        else:   # elif self.mode == "cube":
            max_corner = pcl.max(dim=-2, keepdim=True).values   # (..., 1, K)
            min_corner = pcl.min(dim=-2, keepdim=True).values   # (..., 1, K)
            self.translation = (max_corner + min_corner) / 2    # (..., 1, K)
            unit_scale = (max_corner - min_corner).max(         # (..., 1, 1)
                dim=-1, keepdim=True
            ).values
            # `self.target_scale * 2` because `self.target_scale` is the half
            # side length of the target bounding cube
            self.scale = unit_scale / (self.target_scale * 2)   # (..., 1, 1)

    def apply_same(self, pcl):
        """
        Args:
            pcl (torch.Tensor): Tensor of K-D point clouds of size P with 
                                shape (..., P, K)
        """
        return (pcl - self.translation) / self.scale            # (..., P, K)

    def apply_same_inverse(self, pcl):
        """
        Args:
            pcl (torch.Tensor): Tensor of K-D point clouds of size P with 
                                shape (..., P, K)
        """
        return (pcl * self.scale) + self.translation            # (..., P, K)


class DirectionVectorNoise(Transform):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def _before_apply(self, dirvec):
        # TODO: Refer `GaussianNoise`
        raise NotImplementedError

    def apply_same(self, dirvec):
        raise NotImplementedError
