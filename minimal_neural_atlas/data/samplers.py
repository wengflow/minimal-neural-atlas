import torch


class UniformSampler(torch.utils.data.IterableDataset):
    def __init__(
        self,
        low,
        high,
        size
    ):
        super().__init__()
        self.low = low
        self.high = high
        self.size = size

    def __iter__(self):
        while True:
            yield (self.high - self.low) * torch.rand(self.size) + self.low


class RegularSampler:
    def __init__(
        self,
        low,
        high
    ):
        """
        Args:
            low (Sequence): Lower bound values of each dimension of the
                            regular samples, inclusive (1D sequence of length
                            D)
            high (Sequence): Upper bound values of each dimension of the
                             regular samples, inclusive (1D sequence of
                             length D)
        """
        self.low = low
        self.high = high

    def __call__(self, size, device):
        """
        Args:
            size (Sequence of int): Size of each dimension of the regular
                                    samples (1D sequence of length D)
            device (torch.device): Device where the regular samples are to be
                                   located
        Returns:
            samples (torch.Tensor): Tensor of regular samples of shape
                                    (*size, D)
        """
        coordinates = [
            torch.linspace(dim_low, dim_high, dim_size, device=device)
            for dim_low, dim_high, dim_size in zip(self.low, self.high, size)
        ]
        samples_tuple = torch.meshgrid(*coordinates)
        samples = torch.stack(samples_tuple, dim=-1)
        return samples
