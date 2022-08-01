import easydict
import torch
import pytorch3d.ops


class Loss(torch.nn.Module):
    LOSS_NAMES = [
        "chamfer_dist",
        "occupancy",
        "distortion"
    ]

    def __init__(
        self,
        num_charts,
        sigmoid_temperature,
        distortion_eps,
        loss_weight
    ):
        super().__init__()
        assert isinstance(sigmoid_temperature, (int, float)) \
               and sigmoid_temperature > 0
        assert isinstance(distortion_eps, float) and distortion_eps >= 0

        # add the constant 1.0 weight for the chamfer distance loss
        loss_weight.chamfer_dist = 1.0

        # assert loss names & weights
        assert set(loss_weight.keys()) == set(self.LOSS_NAMES)
        for loss_weight_value in loss_weight.values():
            assert isinstance(loss_weight_value, (int, float)) \
                   and loss_weight_value >= 0

        # save some hyperparameters as attributes
        self.num_charts = num_charts
        self.distortion_eps = distortion_eps
        self.loss_weight = loss_weight

        # save some (derived) loss hyperparameters as buffers
        self.register_buffer(
            "sigmoid_temperature", torch.tensor(sigmoid_temperature)
        )

    def init_batch_mean_loss(self, device):
        batch_mean_loss = easydict.EasyDict({
            loss_name: torch.as_tensor(0., device=device)
            for loss_name in self.LOSS_NAMES
            if self.loss_weight[loss_name] > 0
        })
        return batch_mean_loss

    def compute(
        self,
        batch_true_padded_input_uv,                                             # (num_charts, batch.size, batch.chart_uv_sample_size, 3)
        batch_signed_dist,                                                      # (num_charts, batch.size, batch.chart_uv_sample_size)
        batch_mapped_pcl,                                                       # (num_charts, batch.size, batch.chart_uv_sample_size, 3)
        batch_dataset_target_pcl,                                               # (batch.size, train/eval_target_pcl_nml_size, 3)
        batch_mapped_tangent_vectors=None                                       # (num_charts, batch.size, batch.chart_uv_sample_size, 2, 3)
    ):
        """
        NOTE:
            Care must be taken to handle the computation of losses with
            zero-shaped inputs.
        """
        batch_mean_loss = easydict.EasyDict({})

        # transpose the inputs such that the batch dimension comes first
        batch_true_padded_input_uv = batch_true_padded_input_uv.transpose(0, 1) # (batch_size, num_charts, batch.chart_uv_sample_size, 3)
        batch_signed_dist = batch_signed_dist.transpose(0, 1)                   # (batch_size, num_charts, batch.chart_uv_sample_size)
        batch_mapped_pcl = batch_mapped_pcl.transpose(0, 1)                     # (batch_size, num_charts, batch.chart_uv_sample_size, 3)
        if self.loss_weight.distortion > 0:
            batch_mapped_tangent_vectors = (                                    # (batch.size, num_charts, batch.chart_uv_sample_size, 2, 3)
                batch_mapped_tangent_vectors.transpose(0, 1)
            )

        # compute the chamfer distance
        batch_mean_loss.chamfer_dist, target_pcl_mapped_nn_idx = (              # (), (batch.size, train/eval_target_pcl_nml_size)
            self.chamfer_dist(
                batch_mapped_pcl, batch_dataset_target_pcl
            )
        )

        # derive the mask that indicates whether the mapped points are nearest
        # neighbors of the target point cloud
        batch_chart_uv_sample_size = batch_signed_dist.shape[2]
        is_target_pcl_mapped_nn = self.derive_is_target_pcl_mapped_nn(          # (batch_size, num_charts, batch_chart_uv_sample_size)
            target_pcl_mapped_nn_idx, batch_chart_uv_sample_size
        )

        # compute the number of unique (mapped point cloud) nearest neighbors
        # of the target point cloud, that is associated to each chart
        num_unique_target_nn = is_target_pcl_mapped_nn.sum(dim=2).to(           # (batch.size, num_charts)
            torch.get_default_dtype()
        )
        std_mean_num_unique_target_nn = torch.std_mean(                         # 2-tuple of (batch.size) tensor
            num_unique_target_nn, dim=1, unbiased=False
        )
        std_mean_num_unique_target_nn = torch.stack(                            # (batch.size, 2)
            std_mean_num_unique_target_nn, dim=1
        )

        # employ the target point cloud nearest neighbor mask as the pseudo
        # target occupancies & compute the occupancy loss, if required
        if self.loss_weight.occupancy > 0:
            batch_mean_loss.occupancy = self.occupancy(
                batch_signed_dist, is_target_pcl_mapped_nn
            )

        # compute the distortion loss, if required
        if self.loss_weight.distortion > 0:
            # `batch_mapped_metric_tensor[*indices, :, :]` is the 2x2 metric
            # tensor of the conditional homeomorphism, at
            # `batch.true_padded_input_uv[*indices, :2]`
            batch_mapped_metric_tensor = (                                      # (batch.size, num_charts, batch.chart_uv_sample_size, 2, 2)
                batch_mapped_tangent_vectors 
                @ batch_mapped_tangent_vectors.transpose(3, 4)
            )
            batch_mean_loss.distortion, optimal_sim_scale = self.distortion(    # (batch.size), ()
                batch_mapped_metric_tensor, is_target_pcl_mapped_nn,
                self.distortion_eps
            )
            std_mean_optimal_sim_scale = torch.std_mean(                        # 2-tuple of () tensor
                optimal_sim_scale, unbiased=False
            )
            std_mean_optimal_sim_scale = torch.stack(                           # (2)
                std_mean_optimal_sim_scale, dim=0
            )
        else:
            std_mean_optimal_sim_scale = None

        return batch_mean_loss, std_mean_num_unique_target_nn, \
               std_mean_optimal_sim_scale

    def signed_dist_to_prob_occ(self, signed_dist):
        return torch.sigmoid(signed_dist / self.sigmoid_temperature)

    def prob_occ_to_signed_dist(self, prob_occ):
        return torch.logit(prob_occ) * self.sigmoid_temperature

    @staticmethod
    def chamfer_dist(
        batch_mapped_pcl,                                                       # (batch.size, num_charts, batch.chart_uv_sample_size, 3)
        batch_dataset_target_pcl                                                # (batch.size, train/eval_target_pcl_nml_size, 3)
    ):
        # reshape the mapped point clouds
        batch_size = len(batch_dataset_target_pcl)
        batch_mapped_pcl = batch_mapped_pcl.view(batch_size, -1, 3)             # (batch_size, num_charts * batch.chart_uv_sample_size, 3)

        # compute the Chamfer distance wrt. the mapped point cloud, for each
        # point in the target point cloud
        target_pcl_mapped_nn_dists, target_pcl_mapped_nn_idx, _ = (             # (batch.size, train/eval_target_pcl_nml_size, 1), (batch.size, train/eval_target_pcl_nml_size, 1)
            pytorch3d.ops.knn_points(                                           # (1, train/eval_target_pcl_nml_size, 1), (1, train/eval_target_pcl_nml_size, 1)
                batch_dataset_target_pcl,                                       # (batch.size, train/eval_target_pcl_nml_size, 3)
                batch_mapped_pcl,                                               # (batch_size, num_charts * batch.chart_uv_sample_size, 3)
                K=1
            )
        )
        target_pcl_mapped_nn_dists = target_pcl_mapped_nn_dists[..., 0]         # (batch.size, train/eval_target_pcl_nml_size)
        target_pcl_mapped_nn_idx = target_pcl_mapped_nn_idx[..., 0]             # (batch.size, train/eval_target_pcl_nml_size)
        
        target_pcl_cd = target_pcl_mapped_nn_dists                              # (batch.size, train/eval_target_pcl_nml_size)
        chamfer_dist = target_pcl_cd.mean()
        return chamfer_dist, target_pcl_mapped_nn_idx                           # (), (batch.size, train/eval_target_pcl_nml_size)

    def derive_is_target_pcl_mapped_nn(
        self,
        target_pcl_mapped_nn_idx,                                               # (batch.size, train/eval_target_pcl_nml_size)
        batch_chart_uv_sample_size
    ):                      
        batch_size = len(target_pcl_mapped_nn_idx)
        ones = torch.ones_like(                                                 # (batch_size, train/eval_target_pcl_nml_size)
            target_pcl_mapped_nn_idx, dtype=torch.bool
        )
        is_target_pcl_mapped_nn = torch.zeros(                                  # (batch_size, num_charts * batch_chart_uv_sample_size)
            (batch_size, self.num_charts * batch_chart_uv_sample_size),
            dtype=torch.bool, device=target_pcl_mapped_nn_idx.device
        )
        is_target_pcl_mapped_nn.scatter_(                                       # (batch_size, num_charts * batch_chart_uv_sample_size)
            dim=1, index=target_pcl_mapped_nn_idx, src=ones
        )
        is_target_pcl_mapped_nn = is_target_pcl_mapped_nn.view(                 # (batch_size, num_charts, batch_chart_uv_sample_size)
            batch_size, self.num_charts, batch_chart_uv_sample_size
        )
        return is_target_pcl_mapped_nn

    def occupancy(
        self,
        signed_dist,                                                            # (batch.size, num_charts, batch.chart_uv_sample_size)
        is_target_pcl_mapped_nn                                                 # (batch.size, num_charts, batch.chart_uv_sample_size)
    ):
        """
        NOTE:
            Computing the log probabilistic occupancy directly from the
            probabilistic occupancy is not numerically stable.
        """
        # compute the binary cross entropy occupancy loss
        pseudo_target_occ = is_target_pcl_mapped_nn.to(signed_dist.dtype)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            signed_dist / self.sigmoid_temperature, pseudo_target_occ
        )

    @staticmethod
    def distortion(
        batch_mapped_metric_tensor,                                             # (batch.size, num_charts, batch.chart_uv_sample_size, 2, 2)
        is_valid,                                                               # (batch_size, num_charts, batch.chart_uv_sample_size)
        distortion_eps
    ):
        eye = torch.eye(                                                        # (2, 2)
            2,
            dtype=batch_mapped_metric_tensor.dtype,
            device=batch_mapped_metric_tensor.device
        )
        conditioned_metric_tensor = (                                           # (batch.size, num_charts, batch.chart_uv_sample_size, 2, 2)
            batch_mapped_metric_tensor + distortion_eps * eye
        )

        # compute the forward & backward Dirichlet energies
        """
        NOTE:
            1. Absolute value of the conditioned metric tensor determinant is
               necessary to prevent negative backward Dirichlet energy leading
               to NaN scaled symmetric Dirichlet energy, as a result of
               numerical stability issues.
            2. `Loss.det_two_by_two()` is used instead of `torch.det()` due to
               random illegal memory access runtime errors.
        """
        conditioned_metric_tensor_trace = Loss.trace(conditioned_metric_tensor) # (batch.size, num_charts, batch.chart_uv_sample_size)
        forward_dirichlet_energy = conditioned_metric_tensor_trace              # (batch.size, num_charts, batch.chart_uv_sample_size)
        # the backward Dirichlet energy is essentially 
        # `Loss.trace(conditioned_metric_tensor.inverse())`
        backward_dirichlet_energy = conditioned_metric_tensor_trace \
                                    / torch.abs(Loss.det_two_by_two(
                                        conditioned_metric_tensor
                                    ))                                          # (batch.size, num_charts, batch.chart_uv_sample_size)

        # compute the mean forward & backward Dirichlet energies associated to
        # mapped tangent vectors that are valid, for each sample in the batch
        num_target_pcl_mapped_nn = is_valid.sum(dim=(1, 2))                     # (batch.size)
        forward_dirichlet_energy = torch.sum(                                   # (batch.size)
            torch.where(
                is_valid,
                forward_dirichlet_energy,
                torch.tensor(0.0, dtype=forward_dirichlet_energy.dtype,
                             device=forward_dirichlet_energy.device)
            ), dim=(1, 2)
        ) / num_target_pcl_mapped_nn
        backward_dirichlet_energy = torch.sum(                                  # (batch.size)
            torch.where(
                is_valid,
                backward_dirichlet_energy,
                torch.tensor(0.0, dtype=backward_dirichlet_energy.dtype,
                             device=backward_dirichlet_energy.device)
            ), dim=(1, 2)
        ) / num_target_pcl_mapped_nn

        # infer the global scale that yields the optimal scaled symmetric
        # Dirichlet energy, for each sample in the batch
        """
        NOTE:
            Absolute value of the squared optimal scale is necessary to prevent
            negative values leading to NaN optimal scale, as a result of
            numerical stability issues.
        """
        optimal_sim_scale = torch.sqrt(torch.abs(                               # (batch.size)
            (forward_dirichlet_energy / backward_dirichlet_energy).sqrt()
            - distortion_eps
        ))

        # compute the mean scaled symmetric Dirichlet energy, for each sample
        scaled_symmetric_dirichlet_energy = 2 * torch.sqrt(                     # (batch.size)
            forward_dirichlet_energy * backward_dirichlet_energy
        )

        # compute the scaled symmetric Dirichlet energy averaged across all 
        # samples, with the minimum offset value of 4 subtracted
        MIN_SCALED_SYMMETRIC_DIRICHLET_ENERGY = 4.0
        mean_scaled_symmetric_dirichlet_energy = (
            scaled_symmetric_dirichlet_energy.mean()
            - MIN_SCALED_SYMMETRIC_DIRICHLET_ENERGY
        )

        return mean_scaled_symmetric_dirichlet_energy, optimal_sim_scale        # (), (batch.size)

    @staticmethod
    def trace(tensor):
        """Batch compute the trace of a tensor
        Args:
            tensor (torch.Tensor): Tensor of shape (..., N, N)
        Returns:
            trace (torch.Tensor): Trace of shape (...)
        """
        return tensor.diagonal(dim1=-1, dim2=-2).sum(dim=-1)

    @staticmethod
    def det_two_by_two(tensor):
        """Batch compute the determinant of a tensor of 2x2 matrices
        Args:
            tensor (torch.Tensor): Tensor of shape (..., 2, 2)
        Returns:
            det_two_by_two (torch.Tensor): Determinant of shape (...)
        """
        return (
            tensor[..., 0, 0] * tensor[..., 1, 1]
            - tensor[..., 0, 1] * tensor[..., 1, 0]
        )
