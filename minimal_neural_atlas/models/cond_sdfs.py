import math
import torch
from . import encoders
from ..utils import modules


# TODO: CondSdf mixin
class MlpCondSdf(torch.nn.Module):
    """
    Inspired by:
        1. "DeepSDF: Learning Continuous Signed Distance Functions for Shape
           Representation"
        2. "Multiview Neural Surface Reconstruction by Disentangling Geometry
           and Appearance"
        3. "Neural Lumigraph Rendering"
        4. "Learning Implicit Fields for Generative Shape Modeling"
    """

    def __init__(
        self,
        latent_dims,
        num_pos_encoding_octaves,
        num_hidden_layers,
        hidden_layer_dims,
        hidden_activation,
        output_activation,
        weight_norm,
        concat_input,
        euclidean_space_dims,
        euclidean_space_scale
    ):
        super().__init__()
        assert isinstance(latent_dims, int) and latent_dims >= 0
        assert isinstance(num_pos_encoding_octaves, int) \
               or num_pos_encoding_octaves is None
        assert num_hidden_layers > 0
        assert isinstance(weight_norm, bool)
        assert euclidean_space_dims > 0
        assert isinstance(euclidean_space_scale, (int, float)) \
               and euclidean_space_scale > 0.

        # save some derived arguments as attributes
        self.requires_pos_encoding = num_pos_encoding_octaves is not None
        self.num_fc_layers = num_hidden_layers + 1  # + 1 output layer
        self.input_concat_layer_indices = {
            "all": list(range(1, self.num_fc_layers - 1)),
            "midway": [ self.num_fc_layers // 2 ],
            None: []
        }[concat_input] # "all": all layers except input & last hidden [Ref. 4]

        # instantiate the positional encoding layer, if required
        if self.requires_pos_encoding:
            self.positional_encoder = encoders.PositionalEncoder(
                num_pos_encoding_octaves,
                euclidean_space_scale,
                include_input=True
            )
            pos_encoding_dims = euclidean_space_dims \
                                * self.positional_encoder.dims
            input_dims = pos_encoding_dims + latent_dims
        else:
            input_dims = euclidean_space_dims + latent_dims

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
                out_features = 1                                    # 1 signed distance val
            
            self.fc_layers[fc_layer_index] = torch.nn.Linear(
                in_features, out_features
            )
        self.fc_layers = torch.nn.ModuleList(self.fc_layers)

        # References:
        #   https://github.com/vsitzmann/siren/blob/master/modules.py#L50
        if hidden_activation == "siren":
            raise NotImplementedError   # TODO
        
        # instantiate activations
        self.hidden_activation = {
            "softplus": torch.nn.Softplus(beta=100),
            "relu": torch.nn.ReLU()
            # "siren":  # TODO
        }[hidden_activation]

        max_distance = 2 * math.sqrt(euclidean_space_dims) \
                       * euclidean_space_scale
        self.output_activation = {
            "tanh": modules.ScaledTanh(max_distance),
            None: None
        }[output_activation]

        # manually initialize weights of fully connected layers, if SIREN
        # hidden activation is used
        if hidden_activation == "siren":
            pass    # TODO

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
                                ([N,] P, euclidean_space_dims)
        Returns:
            signed_distance: Signed distances of shape ([N,] P) corresponding
                             to `pcl`
        """
        # expand the latent code across the P dimension
        latent_code = latent_code.unsqueeze(-2)         # ([N,] 1, latent_dims)
        expanded_shape = list(latent_code.shape)
        expanded_shape[-2] = pcl.shape[-2]              # ie. [N,] P, latent_dims
        latent_code = latent_code.expand(               # ([N,] P, latent_dims)
            *expanded_shape
        )
        
        # apply positional encoding to the point cloud, if required
        if self.requires_pos_encoding:
            pcl = self.positional_encoder(pcl)          # ([N,] P, euclidean_space_dims, self.positional_encoder.dims)
            pos_encoded_shape = list(pcl.shape)[:-1]
            pos_encoded_shape[-1] = pcl.shape[-2] * pcl.shape[-1]
            pcl = pcl.view(*pos_encoded_shape)          # ([N,] P, euclidean_space_dims * self.positional_encoder.dims / pos_encoding_dims)

        # concatenate the expanded latent code & (positional-encoded)
        # point cloud to give the input of `self.fc_layers`
        input = torch.cat((latent_code, pcl), dim=-1)   # ([N,] P, latent_dims + pos_encoding_dims/euclidean_space_dims)

        # pass the input through `self.fc_layers`
        data = input
        for fc_layer_index, fc_layer in enumerate(self.fc_layers):
            if fc_layer_index in self.input_concat_layer_indices:
                data = torch.cat((data, input), dim=-1) # ([N,], P, hidden_layer_dims + latent_dims + pos_encoding_dims/euclidean_space_dims)
            data = fc_layer(data)                       # ([N,], P, fc_layer.out_features)

            if fc_layer_index < self.num_fc_layers - 1: # ie. not last fc layer
                data = self.hidden_activation(data)
            elif self.output_activation is not None:    # and is last fc layer
                data = self.output_activation(data)     # ([N,], P, 1)

        signed_distance = data.squeeze(-1)              # ([N,], P)
        return signed_distance
