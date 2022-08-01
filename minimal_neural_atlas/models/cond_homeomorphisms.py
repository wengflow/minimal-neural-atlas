import torch
from ..utils import modules


# TODO: CondHomeomorphism mixin
class MlpCondHomeomorphism(torch.nn.Module):
    """
    Inspired by:
        1. "DeepSDF: Learning Continuous Signed Distance Functions for Shape
           Representation"
        2. "Multiview Neural Surface Reconstruction by Disentangling Geometry
           and Appearance"
        3. "Learning Implicit Fields for Generative Shape Modeling"
        4. "FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation"

    NOTE:
        1. When `reverse=False`, this module only models the (conditional)
           homeomorphism from the `euclidean_space_dims-1`-D Euclidean space to
           a `euclidean_space_dims-1`-D manifold embedded in `euclidean_space_
           dims`-D Euclidean space, although the input point cloud `pcl` is of
           `euclidean_space_dims`-D (ie. the last coordinate is assumed to be 0
           throughout `pcl`), for computational efficiency reasons.
        2. When `reverse=True`, this module only models the (conditional)
           homeomorphism from the `euclidean_space_dims-1`-D manifold embedded
           in `euclidean_space_dims`-D Euclidean space to a `euclidean_space
           _dims-1`-D Euclidean space, although the output/deformed point cloud
           is of `euclidean_space_dims`-D (ie. the last coordinate is set to 0
           throughout the output/deformed point cloud), in order to restrict
           the manifold to `euclidean_space_dims-1`-D.
    """

    def __init__(
        self,
        latent_dims,
        num_hidden_layers,
        hidden_layer_dims,
        hidden_activation,
        output_activation,
        weight_norm,
        concat_input,
        euclidean_space_dims,
        euclidean_space_scale,
        reverse=False
    ):
        super().__init__()
        assert isinstance(latent_dims, int) and latent_dims >= 0
        assert num_hidden_layers > 0
        assert isinstance(weight_norm, bool)
        assert euclidean_space_dims > 0
        assert isinstance(euclidean_space_scale, (int, float)) \
               and euclidean_space_scale > 0.
        assert isinstance(reverse, bool)

        # save some (derived) arguments as attributes
        self.reverse = reverse
        self.num_fc_layers = num_hidden_layers + 1  # + 1 output layer
        self.input_concat_layer_indices = {
            "all": list(range(1, self.num_fc_layers - 1)),
            "midway": [ self.num_fc_layers // 2 ],
            None: []
        }[concat_input] # "all": all layers except input & last hidden [Ref. 4]

        # define the input & output dimensions based on the direction of the
        # (conditional) homeomorphism
        if not reverse:                                             # refer to note 1
            input_dims = (euclidean_space_dims - 1) + latent_dims
            output_dims = euclidean_space_dims
        else:                                                       # refer to note 2
            input_dims = euclidean_space_dims + latent_dims
            output_dims = euclidean_space_dims - 1

        # instantiate and compile the fully connected layers
        self.fc_layers = [ None ] * self.num_fc_layers
        for fc_layer_index in range(self.num_fc_layers):
            in_features = hidden_layer_dims                         # other hidden layers
            out_features = hidden_layer_dims

            if fc_layer_index == 0:                                 # first hidden layer
                in_features = input_dims
            if fc_layer_index in self.input_concat_layer_indices:   # input concat layer
                in_features = hidden_layer_dims + input_dims
            if fc_layer_index == self.num_fc_layers - 1:            # output layer
                out_features = output_dims
            
            self.fc_layers[fc_layer_index] = torch.nn.Linear(
                in_features, out_features
            )
        self.fc_layers = torch.nn.ModuleList(self.fc_layers)
        
        # instantiate activations
        self.hidden_activation = {
            "softplus": torch.nn.Softplus(beta=100),
            "relu": torch.nn.ReLU()
        }[hidden_activation]

        self.output_activation = {
            "tanh": modules.ScaledTanh(euclidean_space_scale),
            None: None
        }[output_activation]

        # apply weight norm, if required
        if weight_norm:
            for fc_layer_index in range(self.num_fc_layers):
                self.fc_layers[fc_layer_index] = torch.nn.utils.weight_norm(
                    self.fc_layers[fc_layer_index]
                )

    def forward(self, latent_code, pcl):
        """
        Args:
            latent_code (torch.Tensor): Latent code of shape ([N,] latent_dims)
            pcl (torch.Tensor): `euclidean_space_dims`-D point cloud of shape
                                ([N,] P, euclidean_space_dims). When `reverse
                                =False`, the last coordinate is always 0 (it is
                                essentially a `euclidean_space_dims-1`-D point
                                cloud).
        Returns:
            deformed_pcl: `euclidean_space_dims`-D point cloud of shape
                          ([N,] P, euclidean_space_dims) homeomorphic to `pcl`.
                          When `reverse=True`, the last coordinate is always 0
                          (it is essentially a `euclidean_space_dims-1`-D point
                          cloud).
        """
        # expand the latent code across the P dimension
        latent_code = latent_code.unsqueeze(-2)         # ([N,] 1, latent_dims)
        expanded_shape = list(latent_code.shape)
        expanded_shape[-2] = pcl.shape[-2]              # ie. [N,] P, latent_dims
        latent_code = latent_code.expand(               # ([N,] P, latent_dims)
            *expanded_shape
        )

        # remove the redundant last coordinate of the point cloud, if the
        # (conditional) homeomorphism is in the forward direction
        if not self.reverse:
            pcl = pcl[..., :-1]                         # ([N,] P, euclidean_space_dims - 1)

        # concatenate the expanded latent code & point cloud to give the input
        # of `self.fc_layers`
        input = torch.cat((latent_code, pcl), dim=-1)   # ([N,] P, input_dims)

        # pass the input through `self.fc_layers`
        data = input
        for fc_layer_index, fc_layer in enumerate(self.fc_layers):
            if fc_layer_index in self.input_concat_layer_indices:
                data = torch.cat((data, input), dim=-1) # ([N,], P, hidden_layer_dims + input_dims)
            data = fc_layer(data)                       # ([N,], P, fc_layer.out_features)

            if fc_layer_index < self.num_fc_layers - 1: # ie. not last fc layer
                data = self.hidden_activation(data)
            elif self.output_activation is not None:    # and is last fc layer
                data = self.output_activation(data)     # ([N,], P, output_dims)

        deformed_pcl = data                             # ([N,], P, output_dims)
        
        # pad the output / deformed point cloud, if the (conditional)
        # homeomorphism is in the reverse direction
        if self.reverse:
            zeros_shape = (*deformed_pcl.shape[:-1], 1) # ie. ([N,], P, 1)
            zeros = torch.zeros(                        # ([N,], P, 1)
                zeros_shape,
                dtype=deformed_pcl.dtype,
                device=deformed_pcl.device
            )
            deformed_pcl = torch.cat(                   # ([N,], P, euclidean_space_dims / output_dims + 1)
                [ deformed_pcl, zeros ], dim=-1
            )
        return deformed_pcl
