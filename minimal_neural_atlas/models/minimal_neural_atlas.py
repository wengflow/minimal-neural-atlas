import copy
import math
import easydict
import torch
import torchvision
import pytorch_lightning as pl
import pytorch3d.structures

from . import cond_sdfs, cond_homeomorphisms, encoders
from .. import loss_metric
from ..data import samplers
from ..utils import autograd, modules


class MinimalNeuralAtlas(pl.LightningModule):
    def __init__(
        self,
        git_head_hash,
        input,
        target,
        num_charts,
        train_uv_sample_size,
        eval_uv_presample_size,
        min_interior_ratio,
        prob_occ_threshold,
        checkpoint_filepath,
        encoder,
        cond_sdf,
        cond_homeomorphism,
        loss,
        metric,
        optimizer,
        lr_scheduler,
        uv_space_scale,
        pcl_normalization_scale,
        train_uv_max_sample_size,
        eval_uv_max_sample_size,
        train_target_pcl_nml_size,
        eval_target_pcl_nml_size
    ):
        super().__init__()
        assert train_target_pcl_nml_size <= train_uv_sample_size \
                <= train_uv_max_sample_size
        assert eval_uv_presample_size <= eval_target_pcl_nml_size \
               <= eval_uv_max_sample_size
        assert min_interior_ratio > 0
        assert 0 <= prob_occ_threshold <= 1.0

        for model_component_config in [
            encoder, cond_sdf, cond_homeomorphism
        ]:
            assert isinstance(model_component_config.load_state_dict, bool)
            assert isinstance(model_component_config.freeze, bool)
            # parameters & buffers of a model component can only be frozen, if
            # it is loaded from a checkpoint
            if model_component_config.freeze:
                assert model_component_config.load_state_dict

        # save some non-hyperparameter configs or its derivatives as attributes
        self.input_set = set(input)
        self.target = target
        self.uv_space_scale = uv_space_scale
        self.pcl_normalization_scale = pcl_normalization_scale

        self.chart_uv_max_sample_size = easydict.EasyDict({
            "train": train_uv_max_sample_size // num_charts,
            "val": eval_uv_max_sample_size // num_charts,
            "test": eval_uv_max_sample_size // num_charts
        })
        self.chart_target_pcl_nml_size = easydict.EasyDict({
            "train": train_target_pcl_nml_size // num_charts,
            "val": eval_target_pcl_nml_size // num_charts,
            "test": eval_target_pcl_nml_size // num_charts
        })

        # log, checkpoint & save hyperparameters to `hparams` attribute
        self.save_hyperparameters(
            "git_head_hash",
            "num_charts",
            "train_uv_sample_size",
            "eval_uv_presample_size",
            "min_interior_ratio",
            "prob_occ_threshold",
            "checkpoint_filepath",
            "encoder",
            "cond_sdf",
            "cond_homeomorphism",
            "loss",
            "metric",
            "optimizer",
            "lr_scheduler",
        )

        # cache some derived hyperparameters as attributes
        self.train_chart_uv_sample_size = train_uv_sample_size // num_charts
        self.eval_chart_uv_presample_size = eval_uv_presample_size \
                                            // num_charts

        # cache some mesh & point cloud log configs as attributes
        self.has_logged_target_pcl = False
        self.encoded_mesh_log_config, self.mapped_mesh_log_config = (
            self._build_mesh_log_config()
        )

        # instantiate model components
        self.encoder = self._build_encoder()
        self.latent_reductions = self._build_latent_reductions()
        self.cond_sdfs = self._build_cond_sdfs()
        self.cond_homeomorphisms = self._build_cond_homeomorphisms()

        # load parameters & buffers of model components from a checkpoint &
        # freeze them, if required
        self._load_model_component_state_dicts()
        self._freeze_model_components()

        # instantiate regular sampler, loss and metric components
        UV_SAMPLE_DIMS = 2
        self.regular_sampler = samplers.RegularSampler(
            low=[ -uv_space_scale ] * UV_SAMPLE_DIMS,
            high=[ uv_space_scale ] * UV_SAMPLE_DIMS,
        )
        self.loss = loss_metric.loss.Loss(
            num_charts, loss.sigmoid_temperature,
            loss.distortion_eps, loss.weight
        )
        f_score_dist_threshold = metric.f_score_scale_ratio \
                                 * pcl_normalization_scale
        self.metric = loss_metric.metric.Metric(
            num_charts,
            uv_space_scale,
            eval_target_pcl_nml_size,
            metric.default_chamfer_dist,
            metric.default_f_score,
            metric.default_distortion,
            f_score_dist_threshold,
            metric.distortion_eps,
            metric.degen_chart_area_ratio
        )

    @staticmethod
    def _build_mesh_log_config():
        encoded_mesh_log_config = {
            "lights": [
                {
                    "cls": "AmbientLight",
                    "color": "#ffffff",
                    "intensity": 0.75
                },
                {
                    "cls": "DirectionalLight",
                    "color": "#ffffff",
                    "intensity": 0.75,
                    "position": [0, -1, 2]
                },
                {
                    "cls": "DirectionalLight",
                    "color": "#ffffff",
                    "intensity": 0.75,
                    "position": [0, 1, -2]
                }
            ],
            "material": {
                "cls": "MeshStandardMaterial",
                "side": 2   # THREE.DoubleSide
            }
        }
        mapped_mesh_log_config = copy.deepcopy(encoded_mesh_log_config)
        mapped_mesh_log_config["material"]["wireframe"] = True

        return encoded_mesh_log_config, mapped_mesh_log_config

    def _build_encoder(self):
        if len(self.input_set) == 0:  # overfit to a single shape instance
            assert self.hparams.encoder.latent_dims == 0
            assert self.hparams.cond_sdf.reduced_latent_dims is None
            assert self.hparams.cond_homeomorphism.reduced_latent_dims is None
            encoder = None
        elif self.input_set == { "pcl" }:
            encoder = encoders.PointNet(
                self.hparams.encoder.latent_dims,
                self.hparams.encoder.pretrained,
                has_input_normals=False
            )
        elif self.input_set == { "pcl", "nml" }:
            encoder = encoders.PointNet(
                self.hparams.encoder.latent_dims,
                self.hparams.encoder.pretrained,
                has_input_normals=True
            )
        elif self.input_set == { "img" }:
            encoder = encoders.ResNet18(
                self.hparams.encoder.latent_dims,
                self.hparams.encoder.pretrained
            )
        else:
            raise NotImplementedError

        return encoder

    def _build_latent_reductions(self):
        latent_reductions = torch.nn.ModuleDict({})
        if self.hparams.cond_sdf.reduced_latent_dims is not None:
            latent_reductions["cond_sdf"] = torch.nn.ModuleList(
                modules.build_linear_relu(
                    self.hparams.encoder.latent_dims,
                    self.hparams.cond_sdf.reduced_latent_dims
                ) for _ in range(self.hparams.num_charts)
            )
        if self.hparams.cond_homeomorphism.reduced_latent_dims is not None:
            latent_reductions["cond_homeomorphism"] = torch.nn.ModuleList(
                modules.build_linear_relu(
                    self.hparams.encoder.latent_dims,
                    self.hparams.cond_homeomorphism.reduced_latent_dims
                ) for _ in range(self.hparams.num_charts)
            )
        return latent_reductions

    def _build_cond_sdfs(self):
        if self.hparams.cond_sdf.reduced_latent_dims is None:
            cond_sdf_latent_dims = self.hparams.encoder.latent_dims
        else:
            cond_sdf_latent_dims = self.hparams.cond_sdf.reduced_latent_dims

        return torch.nn.ModuleList(
            cond_sdfs.MlpCondSdf(
                cond_sdf_latent_dims,
                self.hparams.cond_sdf.num_pos_encoding_octaves,
                self.hparams.cond_sdf.num_hidden_layers,
                self.hparams.cond_sdf.hidden_layer_dims,
                self.hparams.cond_sdf.hidden_activation,
                self.hparams.cond_sdf.output_activation,
                self.hparams.cond_sdf.weight_norm,
                self.hparams.cond_sdf.concat_input,
                euclidean_space_dims=3,
                euclidean_space_scale=self.pcl_normalization_scale
            )
            for _ in range(self.hparams.num_charts)
        )

    def _build_cond_homeomorphisms(self):
        if self.hparams.cond_homeomorphism.reduced_latent_dims is None:
            cond_homeomorphism_latent_dims = self.hparams.encoder.latent_dims
        else:
            cond_homeomorphism_latent_dims = self.hparams.cond_homeomorphism \
                                                         .reduced_latent_dims

        if self.hparams.cond_homeomorphism.arch == "inv_mlp":
            return torch.nn.ModuleList(
                cond_homeomorphisms.MlpCondHomeomorphism(
                    cond_homeomorphism_latent_dims,
                    self.hparams.cond_homeomorphism.inv_mlp \
                                                    .num_hidden_layers,
                    self.hparams.cond_homeomorphism.inv_mlp \
                                                    .hidden_layer_dims,
                    self.hparams.cond_homeomorphism.inv_mlp \
                                                    .hidden_activation,
                    self.hparams.cond_homeomorphism.inv_mlp \
                                                    .fwd_output_activation,
                    self.hparams.cond_homeomorphism.inv_mlp.weight_norm,
                    self.hparams.cond_homeomorphism.inv_mlp.concat_input,
                    euclidean_space_dims=3,
                    euclidean_space_scale=self.pcl_normalization_scale
                )
                for _ in range(self.hparams.num_charts)
            )
        else:
            raise NotImplementedError

    def _load_model_component_state_dicts(self):
        # return, if none of the model components are required to load
        # parameters & buffers from a checkpoint
        if not (
            self.hparams.encoder.load_state_dict
            or self.hparams.cond_sdf.load_state_dict
            or self.hparams.cond_homeomorphism.load_state_dict
        ):
            return

        checkpoint = torch.load(
            self.hparams.checkpoint_filepath, map_location=torch.device('cpu')
        )
        if self.hparams.encoder.load_state_dict:
            self.encoder.load_state_dict(
                modules.extract_descendent_state_dict(
                    checkpoint["state_dict"], "encoder"
                )
            )
        if self.hparams.cond_sdf.load_state_dict:
            self.cond_sdfs.load_state_dict(
                modules.extract_descendent_state_dict(
                    checkpoint["state_dict"], "cond_sdfs"
                )
            )
            if self.hparams.cond_sdf.reduced_latent_dims is not None:
                self.latent_reductions["cond_sdf"].load_state_dict(
                    modules.extract_descendent_state_dict(
                        checkpoint["state_dict"], "latent_reductions.cond_sdf"
                    )
                )
        if self.hparams.cond_homeomorphism.load_state_dict:
            self.cond_homeomorphisms.load_state_dict(
                modules.extract_descendent_state_dict(
                    checkpoint["state_dict"], "cond_homeomorphisms"
                )
            )
            if self.hparams.cond_homeomorphism.reduced_latent_dims is not None:
                self.latent_reductions["cond_homeomorphism"].load_state_dict(
                    modules.extract_descendent_state_dict(
                        checkpoint["state_dict"],
                        "latent_reductions.cond_homeomorphism"
                    )
                )

    def _freeze_model_components(self):
        if self.hparams.encoder.freeze:
            modules.freeze(self.encoder)
        if self.hparams.cond_sdf.freeze:
            modules.freeze(self.cond_sdfs)
            if self.hparams.cond_sdf.reduced_latent_dims is not None:
                modules.freeze(self.latent_reductions.cond_sdf)
        if self.hparams.cond_homeomorphism.freeze:
            modules.freeze(self.cond_homeomorphisms)
            if self.hparams.cond_homeomorphism.reduced_latent_dims is not None:
                modules.freeze(self.latent_reductions.cond_homeomorphism)

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_index):
        batch = easydict.EasyDict(batch)
        batch.size = len(batch.dataset.target.pcl)

        # convert the unicode code point tensors of class & sample IDs to strs
        batch.dataset.common.class_id = self.unicode_code_pt_tensor_to_str(
            batch.dataset.common.class_id
        )
        batch.dataset.common.sample_id = self.unicode_code_pt_tensor_to_str(
            batch.dataset.common.sample_id
        )

        # infer the latent codes
        batch.latent_code = self.infer_latent_code(batch)
        # remove redundant tensors to attempt to free-up memory
        if len(self.input_set) > 0:
            batch.dataset.pop("input")

        # preprocess the input UV samples
        batch.padded_input_uv = self.preprocess_input_uv(                       # (num_charts, batch.size, train_uv_max_sample_size // num_charts, 3)
            batch.input_uv, batch.size
        )

        # extract the true padded input UV samples
        batch.chart_uv_sample_size = self.train_chart_uv_sample_size
        batch.true_padded_input_uv = (                                          # (num_charts, batch.size, batch.chart_uv_sample_size, 3)
            batch.padded_input_uv[..., :batch.chart_uv_sample_size, :]
        )
        batch.pop("input_uv")                           # free-up memory
        batch.pop("padded_input_uv")

        # infer the mapped pcl associated to the true padded input UV samples
        batch.mapped_pcl = self.infer_pcl(                                      # (num_charts, batch.size, batch.chart_uv_sample_size, 3)
            batch.latent_code.cond_homeomorphism,
            batch.true_padded_input_uv
        )
        batch.mapped_pcl = self.to_batch_size_num_charts_contiguous(            # (num_charts, batch.size, batch.chart_uv_sample_size, 3)
            batch.mapped_pcl
        )

        # infer the signed distances associated to the true padded input UV
        # samples & mapped point cloud
        if self.hparams.loss.decouple_grad_flow:
            batch.signed_dist = self.infer_signed_dist(                         # (num_charts, batch.size, batch.chart_uv_sample_size)
                batch.latent_code.cond_sdf, batch.mapped_pcl.detach()
            )
        else:
            batch.signed_dist = self.infer_signed_dist(                         # (num_charts, batch.size, batch.chart_uv_sample_size)
                batch.latent_code.cond_sdf, batch.mapped_pcl
            )
        batch.signed_dist = self.to_batch_size_num_charts_contiguous(           # (num_charts, batch.size, batch.chart_uv_sample_size)
            batch.signed_dist
        )

        # estimate the label frequency & occupancy ratio
        batch.label_freq = self.estimate_label_freq(                            # (batch.size)
            batch.signed_dist
        )
        batch.is_occupied = self.derive_is_occupied(                            # (num_charts, batch.size, batch.chart_uv_sample_size)
            batch.signed_dist, batch.label_freq
        )
        batch.occupancy_ratio = self.estimate_occupancy_ratio(                  # (batch.size)
            batch.is_occupied
        )

        # infer the tangent vectors associated to the mapped pcl, if required
        if self.hparams.loss.weight.distortion > 0:
            batch.mapped_tangent_vectors = self.infer_mapped_tangent_vectors(   # (num_charts, batch.size, batch.chart_uv_sample_size, 2, 3)
                batch.true_padded_input_uv, batch.mapped_pcl, stage="train"
            )
            batch.mapped_tangent_vectors = (                                    # (num_charts, batch.size, batch.chart_uv_sample_size, 2, 3)
                self.to_batch_size_num_charts_contiguous(           
                    batch.mapped_tangent_vectors
                )
            )
        else:
            batch.mapped_tangent_vectors = None

        # derive the number of partially-occupied charts (ie. non-empty &
        # non-fully-occupied charts)
        batch.chart_is_partial = torch.logical_and(                             # (num_charts, batch.size)
            batch.is_occupied.any(dim=2),
            batch.is_occupied.logical_not().any(dim=2)
        )
        batch.num_partial_charts = (
            batch.chart_is_partial.to(torch.get_default_dtype()).sum(dim=0)     # (batch.size)
        )

        # compute the loss terms
        if "nml" not in self.target:
            batch.dataset.target.nml = None
        batch.mean_loss = self.loss.init_batch_mean_loss(self.device)
        batch.mean_loss_computed, batch.std_mean_num_unique_target_nn, \
        batch.std_mean_optimal_sim_scale = (                                    # (), (batch.size, 2), (2)
            self.loss.compute(
                batch.true_padded_input_uv,
                batch.signed_dist,
                batch.mapped_pcl,
                batch.dataset.target.pcl,
                batch.mapped_tangent_vectors
            )
        )
        batch.mean_loss.update(batch.mean_loss_computed)

        # derive the final loss as the weighted sum of the loss terms
        batch.weighted_mean_loss = easydict.EasyDict({
            loss_name: mean_loss_value * self.hparams.loss.weight[loss_name]
            for loss_name, mean_loss_value in batch.mean_loss.items()
        })
        train_loss = sum(batch.weighted_mean_loss.values())

        # log some quantities for monitoring & debugging
        self.log("train/loss", train_loss, prog_bar=True)
        for loss_name, mean_loss_value in batch.mean_loss.items():
            self.log(f"train/{loss_name}", mean_loss_value)

        batch.mean_label_freq = batch.label_freq.mean()
        batch.std_mean_occupancy_ratio = torch.std_mean(                        # 2-tuple of () tensor
            batch.occupancy_ratio, unbiased=False
        )
        batch.mean_num_partial_charts = batch.num_partial_charts.mean()
        batch.mean_std_mean_num_unique_target_nn = (                            # (2)
            batch.std_mean_num_unique_target_nn.mean(dim=0)
        )
        self.log("train/label_freq", batch.mean_label_freq)
        self.log(
            "train/std_occupancy_ratio", batch.std_mean_occupancy_ratio[0]
        )
        self.log(
            "train/mean_occupancy_ratio", batch.std_mean_occupancy_ratio[1]
        )
        self.log("train/num_partial_charts", batch.mean_num_partial_charts)
        self.log(
            "train/std_num_unique_target_nn", 
            batch.mean_std_mean_num_unique_target_nn[0]
        )
        self.log(
            "train/mean_num_unique_target_nn", 
            batch.mean_std_mean_num_unique_target_nn[1]
        )
        if self.hparams.loss.weight.distortion > 0:
            self.log(
                "train/std_optimal_sim_scale",
                batch.std_mean_optimal_sim_scale[0]
            )
            self.log(
                "train/mean_optimal_sim_scale",
                batch.std_mean_optimal_sim_scale[1]
            )

        return train_loss

    def validation_step(self, batch, batch_index):
        return self.evaluation_step(batch, batch_index, stage="val")

    def test_step(self, batch, batch_index):
        return self.evaluation_step(batch, batch_index, stage="test")

    def evaluation_step(self, batch, batch_index, stage):
        batch = easydict.EasyDict(batch)
        batch.size = len(batch.dataset.target.pcl)

        # convert the unicode code point tensors of class & sample IDs to strs
        batch.dataset.common.class_id = self.unicode_code_pt_tensor_to_str(
            batch.dataset.common.class_id
        )
        batch.dataset.common.sample_id = self.unicode_code_pt_tensor_to_str(
            batch.dataset.common.sample_id
        )

        # preprocess the input UV samples and infer the latent codes in batch
        batch.padded_input_uv = self.preprocess_input_uv(                       # (num_charts, batch.size, eval_uv_max_sample_size // num_charts, 3)
            batch.input_uv, batch.size
        )
        batch.latent_code = self.infer_latent_code(batch)
        # remove redundant tensors to attempt to free-up memory
        batch.pop("input_uv")
        if len(self.input_set) > 0:
            batch.dataset.pop("input")

        # perform the remaining forward passes (& implicitly backward passes)
        # of the conditional SDFs & homeomorphisms for each sample in the
        # batch, due to varying optimal UV sample size
        if batch.size == 1 or self.device.type == "cpu":
            # can also be used for debugging
            batch.stream = [ None ] * batch.size
        elif self.device.type == "cuda":
            batch.stream = [ torch.cuda.Stream(self.device)
                             for _ in range(batch.size) ]
        batch.metric = self.metric.init_batch_metric(batch.size, self.device)
        batch.occupancy_ratio = [ None ] * batch.size
        batch.label_freq = [ None ] * batch.size
        batch.num_partial_charts = [ None ] * batch.size
        batch.prob_occ = [ None ] * batch.size
        batch.encoded_mesh = [ None ] * batch.size
        batch.mapped_mesh = [ None ] * batch.size
        batch.optimal_sim_scale = [ None ] * batch.size
        batch.optimal_area_scale = [ None ] * batch.size
        sample = easydict.EasyDict({})

        if batch.size > 1 and self.device.type == "cuda":
            for sample_stream in batch.stream:
                sample_stream.wait_stream(
                    torch.cuda.current_stream(self.device)
                )
        for sample.index in range(batch.size):
            # parallelize operations, if possible, for each sample in the batch
            sample.stream = batch.stream[sample.index]
            with torch.cuda.stream(sample.stream):
                # extract data that is associated to this sample from the batch
                sample.padded_input_uv = (                                      # (num_charts, eval_uv_max_sample_size // num_charts, 3)
                    batch.padded_input_uv[:, sample.index, ...]
                )
                sample.latent_code = easydict.EasyDict({})
                sample.latent_code.cond_sdf = (                                 # (num_charts, latent_dims / cond_sdf.reduced_latent_dims)
                    batch.latent_code.cond_sdf[:, sample.index, :]
                )
                sample.latent_code.cond_homeomorphism = (                       # (num_charts, latent_dims / cond_homeomorphism.reduced_latent_dims)
                    batch.latent_code.cond_homeomorphism[:, sample.index, :]
                )
                sample.dataset_target = easydict.EasyDict({})
                sample.dataset_target.pcl = (                                   # (eval_target_pcl_nml_size, 3)
                    batch.dataset.target.pcl[sample.index, ...]
                )
                # target surface normal is always given during evaluation
                sample.dataset_target.nml = (                                   # (eval_target_pcl_nml_size, 3)
                    batch.dataset.target.nml[sample.index, ...]
                )

                # presample the conditional homeomorphisms & SDFs to estimate
                # the occupancy ratio
                stop = self.eval_chart_uv_presample_size
                sample.presampling_padded_input_uv = (                          # (num_charts, self.chart_uv_presample_size[stage], 3)
                    sample.padded_input_uv[:, :stop, :]
                )
                sample.presampled_mapped_pcl = self.infer_pcl(                  # (num_charts, self.chart_uv_presample_size[stage], 3)
                    sample.latent_code.cond_homeomorphism,
                    sample.presampling_padded_input_uv
                )
                sample.presampled_signed_dist = self.infer_signed_dist(         # (num_charts, self.chart_uv_presample_size[stage])
                    sample.latent_code.cond_sdf, sample.presampled_mapped_pcl
                )

                sample.presampled_label_freq = self.estimate_label_freq(
                    sample.presampled_signed_dist
                )
                sample.presampled_is_occupied = self.derive_is_occupied(        # (num_charts, self.chart_uv_presample_size[stage])
                    sample.presampled_signed_dist, sample.presampled_label_freq
                )
                sample.presampled_occupancy_ratio = (
                    self.estimate_occupancy_ratio(
                        sample.presampled_is_occupied
                    )
                )
                sample.pop("presampled_mapped_pcl")     # free-up memory
                sample.pop("presampled_signed_dist")
                sample.pop("presampled_is_occupied")

                # derive the optimal perfect square UV sample size for each
                # chart
                sample.chart_uv_sample_size = self.derive_chart_uv_sample_size(
                    sample.presampled_occupancy_ratio, stage=stage
                )
                sample.chart_uv_sample_size_sqrt = (
                    sample.chart_uv_sample_size.sqrt().ceil().to(torch.int64)
                )
                sample.chart_uv_sample_size = (
                    sample.chart_uv_sample_size_sqrt.square()
                )

                # resample the conditional homeomorphisms & SDFs with regular
                # padded UV samples
                sample.true_padded_input_uv = (                                 # (num_charts, sample.chart_uv_sample_size, 3)
                    self.sample_regular_uv(sample.chart_uv_sample_size_sqrt)
                )
                # enable grad to calc. distortion & normal consistency metrics
                with torch.enable_grad():
                    sample.mapped_pcl = self.infer_pcl(                         # (num_charts, sample.chart_uv_sample_size, 3)
                        sample.latent_code.cond_homeomorphism,
                        sample.true_padded_input_uv
                    )
                sample.signed_dist = self.infer_signed_dist(                    # (num_charts, sample.chart_uv_sample_size)
                    sample.latent_code.cond_sdf, sample.mapped_pcl
                )

                # derive the probabilistic occupancy from the signed distances
                # & save it in the batch
                batch.prob_occ[sample.index] = (                                # (num_charts, sample.chart_uv_sample_size)
                    self.loss.signed_dist_to_prob_occ(sample.signed_dist)
                )
                batch.prob_occ[sample.index] = (                                # (num_charts, sample.chart_uv_sample_size_sqrt, sample.chart_uv_sample_size_sqrt)
                    batch.prob_occ[sample.index].view(
                        -1,
                        sample.chart_uv_sample_size_sqrt,
                        sample.chart_uv_sample_size_sqrt
                    )
                )

                # estimate the label frequency & occupancy ratio, then save it
                # in the batch
                sample.label_freq = self.estimate_label_freq(
                    sample.signed_dist
                )
                sample.is_occupied = self.derive_is_occupied(                   # (num_charts, sample.chart_uv_sample_size)
                    sample.signed_dist, sample.label_freq
                )
                sample.occupancy_ratio = self.estimate_occupancy_ratio(
                    sample.is_occupied
                )
                batch.label_freq[sample.index] = sample.label_freq
                batch.occupancy_ratio[sample.index] = sample.occupancy_ratio

                # derive the number of partially-occupied charts (ie. non-empty
                # & non-fully-occupied charts) & save it in the batch
                sample.chart_is_partial = torch.logical_and(                    # (num_charts)
                    sample.is_occupied.any(dim=1),
                    sample.is_occupied.logical_not().any(dim=1)
                )
                batch.num_partial_charts[sample.index] = (
                    sample.chart_is_partial.sum()
                )

                # infer the tangent vectors associated to the mapped pcl
                with torch.enable_grad():
                    sample.mapped_tangent_vectors = (                           # (num_charts, sample.chart_uv_sample_size, 2, 3)
                        self.infer_mapped_tangent_vectors(
                            sample.true_padded_input_uv, sample.mapped_pcl,
                            stage=stage
                        )
                    )

                # compute the metric terms & save them in the batch
                sample.metric, batch.encoded_mesh[sample.index], \
                batch.mapped_mesh[sample.index], \
                batch.optimal_sim_scale[sample.index], \
                batch.optimal_area_scale[sample.index] = self.metric.compute(
                    sample.is_occupied,
                    sample.mapped_pcl,
                    sample.dataset_target.pcl,
                    sample.mapped_tangent_vectors,
                    sample.dataset_target.nml
                )
                for metric_name, metric_value in sample.metric.items():
                    batch.metric[metric_name][sample.index] = metric_value
        if batch.size > 1 and self.device.type == "cuda":
            for sample_stream in batch.stream:
                torch.cuda.current_stream(self.device).wait_stream(
                    sample_stream
                )

        # average the metric terms over the samples of the batch
        batch.mean_metric = easydict.EasyDict({
            metric_name: sum(metric_values) / batch.size
            for metric_name, metric_values in batch.metric.items()
        })

        # log some quantities for monitoring & debugging
        self.log(
            f"{stage}/epoch", self.current_epoch,
            batch_size=batch.size, prog_bar=True, logger=False
        )
        for metric_name, mean_metric_value in batch.mean_metric.items():
            self.log(
                f"{stage}/{metric_name}", mean_metric_value,
                batch_size=batch.size, prog_bar=True
            )

        batch.mean_label_freq = sum(batch.label_freq) / batch.size
        batch.std_mean_occupancy_ratio = torch.std_mean(                        # 2-tuple of () tensor
            torch.as_tensor(batch.occupancy_ratio), unbiased=False
        )
        batch.mean_num_partial_charts = sum(batch.num_partial_charts) \
                                        / batch.size
        batch.std_mean_optimal_sim_scale = torch.std_mean(                      # 2-tuple of () tensor
            torch.as_tensor(batch.optimal_sim_scale), unbiased=False
        )
        batch.std_mean_optimal_area_scale = torch.std_mean(                     # 2-tuple of () tensor
            torch.as_tensor(batch.optimal_area_scale), unbiased=False
        )
        self.log(
            f"{stage}/label_freq", batch.mean_label_freq, batch_size=batch.size
        )
        self.log(
            f"{stage}/std_occupancy_ratio",
            batch.std_mean_occupancy_ratio[0], batch_size=batch.size
        )
        self.log(
            f"{stage}/mean_occupancy_ratio",
            batch.std_mean_occupancy_ratio[1], batch_size=batch.size
        )
        self.log(
            f"{stage}/num_partial_charts",
            batch.mean_num_partial_charts, batch_size=batch.size
        )
        self.log(
            f"{stage}/std_optimal_sim_scale",
            batch.std_mean_optimal_sim_scale[0], batch_size=batch.size
        )
        self.log(
            f"{stage}/mean_optimal_sim_scale",
            batch.std_mean_optimal_sim_scale[1], batch_size=batch.size
        )
        self.log(
            f"{stage}/std_optimal_area_scale",
            batch.std_mean_optimal_area_scale[0], batch_size=batch.size
        )
        self.log(
            f"{stage}/mean_optimal_area_scale",
            batch.std_mean_optimal_area_scale[1], batch_size=batch.size
        )

        # log the probabilistic occupancies of each chart, encoded mesh &
        # mapped mesh, for each sample in the first batch, if this process has
        # global rank of 0
        if self.logger is None or batch_index > 0 or self.global_rank > 0:
            return

        # log the chart probabilistic occupancies
        max_chart_uv_sample_size_sqrt = max(
            sample_prob_occ.shape[1]
            for sample_prob_occ in batch.prob_occ 
        )
        batch.prob_occ_img = [                                                  # batch.size-list of (num_charts, max_chart_uv_sample_size_sqrt, max_chart_uv_sample_size_sqrt) tensor
            torchvision.transforms.functional.resize(
                img=sample_prob_occ,
                size=max_chart_uv_sample_size_sqrt
            )
            for sample_prob_occ in batch.prob_occ
        ]
        batch.prob_occ_img = torch.stack(batch.prob_occ_img, dim=1)             # (num_charts, batch.size, max_chart_uv_sample_size_sqrt, max_chart_uv_sample_size_sqrt)
        
        for chart_index in range(self.hparams.num_charts):
            chart_batch_prob_occ_img = batch.prob_occ_img[chart_index, ...]     # (batch.size, max_chart_uv_sample_size_sqrt, max_chart_uv_sample_size_sqrt)
            self.logger.experiment.add_images(
                f"{stage}/chart_prob_occ/{chart_index}",
                chart_batch_prob_occ_img.unsqueeze(dim=1),                      # (batch.size, 1, max_chart_uv_sample_size_sqrt, max_chart_uv_sample_size_sqrt)
                global_step=self.global_step
            )

        # log the encoded meshes
        batch.encoded_mesh = pytorch3d.structures.join_meshes_as_batch(
            batch.encoded_mesh
        )
        self.logger.experiment.add_mesh(
            f"{stage}/encoded_mesh",
            vertices=batch.encoded_mesh.verts_padded(),
            faces=batch.encoded_mesh.faces_padded(),
            colors=batch.encoded_mesh.textures.verts_features_padded(),
            config_dict=self.encoded_mesh_log_config,
            global_step=self.global_step
        )

        # log the mapped meshes
        batch.mapped_mesh = pytorch3d.structures.join_meshes_as_batch(
            batch.mapped_mesh
        )
        self.logger.experiment.add_mesh(
            f"{stage}/mapped_mesh",
            vertices=batch.mapped_mesh.verts_padded(),
            faces=batch.mapped_mesh.faces_padded(),
            colors=batch.mapped_mesh.textures.verts_features_padded(),
            config_dict=self.mapped_mesh_log_config,
            global_step=self.global_step
        )

        # log the target point cloud, for each sample in the first batch, if
        # it has not been done before & this process has global rank of 0
        if self.has_logged_target_pcl:
            return

        self.logger.experiment.add_mesh(
            f"{stage}/target_pcl",
            vertices=batch.dataset.target.pcl,
            global_step=self.global_step
        )
        self.has_logged_target_pcl = True

    def configure_optimizers(self):
        # instantiate optimizer
        if self.hparams.optimizer.algo == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.optimizer.lr
            )
        else:
            raise NotImplementedError

        # instantiate learning rate scheduler
        if self.hparams.lr_scheduler.algo == "multi_step_lr":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.hparams.lr_scheduler.multi_step_lr.milestones,
                gamma=self.hparams.lr_scheduler.multi_step_lr.gamma
            )
        else:
            raise NotImplementedError
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.hparams.lr_scheduler.interval
            }
        }

    def on_train_start(self):
        if self.logger is None:
            return

        # define the metrics to track for hyperparameter tuning
        self.logger.log_hyperparams(
            self.hparams,
            {
                "val/mesh_chamfer_dist": self.hparams.metric \
                                                     .default_chamfer_dist,
                "val/mesh_f_score": self.hparams.metric.default_f_score,
                "val/encoded_chamfer_dist": self.hparams.metric \
                                                        .default_chamfer_dist,
                "val/encoded_f_score": self.hparams.metric.default_f_score,
                "val/encoded_sim_distortion": self.hparams.metric \
                                                          .default_distortion,
                "val/encoded_conf_distortion": self.hparams.metric \
                                                           .default_distortion,
                "val/encoded_area_distortion": self.hparams.metric \
                                                           .default_distortion,
                "val/num_degenerate_charts": self.hparams.num_charts,
                "val/mean_occupancy_ratio": 0.0,
                "val/std_occupancy_ratio": 0.5,
                "val/num_partial_charts": 0
            }
        )

    def on_train_epoch_start(self):
        """
        NOTE:
            Hotfix for Pytorch Lightning bug on failure to call `set_epoch()`
            on `DistributedSampler` of collection of dataloaders.
        """
        if hasattr(
            self.trainer.train_dataloader.sampler["dataset"], "set_epoch"
        ):
            self.trainer.train_dataloader.sampler["dataset"].set_epoch(
                self.current_epoch
            )

    @staticmethod
    def unicode_code_pt_tensor_to_str(batch_unicode_code_pt_tensor):            # (batch.size, S)
        # for each sample in the batch, convert all Unicode code point integers
        # to characters, then concatenate them to form a string & finally
        # remove any trailing spaces used for padding 
        batch_str = [
            "".join(map(chr, sample_unicode_code_pt_tensor)).rstrip()
            for sample_unicode_code_pt_tensor in batch_unicode_code_pt_tensor
        ]
        return batch_str

    def preprocess_input_uv(self, batch_input_uv, batch_size):
        # remove the extra batch dimension of 1 on the input UV samples
        batch_input_uv = batch_input_uv.squeeze(dim=0)                          # (train/val/test_batch_size, train/eval_uv_max_sample_size, 2)
        
        # match the batch size of the input UV samples to other inputs, which
        # might be different for the last batch during evaluation
        batch_input_uv = batch_input_uv[:batch_size, ...]                       # (batch_size, train/eval_uv_max_sample_size, 2)
        
        # virtually split the input UV samples evenly to each chart
        batch_input_uv = batch_input_uv.view(                                   # (num_charts, batch.size, train/eval_uv_max_sample_size // num_charts, 2)
            self.hparams.num_charts, batch_size, -1, 2
        )

        # pad zeros to the input UV samples as (last) W-coordinates
        # TODO: pad during dataloading
        batch_padded_input_uv = torch.nn.functional.pad(batch_input_uv, (0, 1)) # (num_charts, batch.size, train/eval_uv_max_sample_size // num_charts, 3)

        # require gradients for the padded input UV samples in order to compute
        # gradients of the homeomorphisms
        batch_padded_input_uv.requires_grad_()

        return batch_padded_input_uv

    def infer_latent_code(self, batch):
        latent_code = easydict.EasyDict({})

        # infer the latent code common to all latent code reduction layers, or
        # conditional SDFs & homeomorphisms, if no latent reduction is required              
        if self.input_set == { "pcl" }:
            dataset_input_data = batch.dataset.input.pcl                        # (batch.size, input_pcl_nml_size, 3)
        elif self.input_set == { "pcl", "nml" }:
            dataset_input_data = torch.cat(                                     # (batch.size, input_pcl_nml_size, 6)
                (batch.dataset.input.pcl, batch.dataset.input.nml), dim=-1
            )
        elif self.input_set == { "img" }:
            dataset_input_data = batch.dataset.input.img                        # (batch.size, 3, img_size, img_size)

        if len(self.input_set) == 0:
            latent_code.common = torch.empty(                                   # (batch.size, 0)
                size=(batch.size, 0),
                dtype=batch.dataset.target.pcl.dtype,
                device=batch.dataset.target.pcl.device
            )
        else:
            latent_code.common = self.encoder(dataset_input_data)               # (batch.size, latent_dims)

        # infer chart-specific latent codes for conditional SDF & homeomorphism
        for model_component in [ "cond_sdf", "cond_homeomorphism" ]:
            if self.hparams[model_component].reduced_latent_dims is None:
                latent_code[model_component] = latent_code.common \
                                                          .unsqueeze(dim=0)     # (1, batch.size, latent_dims)
                latent_code[model_component] = (                                # (num_charts, batch.size, latent_dims)
                    latent_code[model_component].expand(
                        self.hparams.num_charts, -1, -1
                    )
                )
            else:   # latent code reduction is required
                latent_code[model_component] = torch.stack([                    # (num_charts, batch.size, cond_sdf/cond_homeomorphism.reduced_latent_dims)
                    chart_model_component_latent_reductions(latent_code.common)
                    for chart_model_component_latent_reductions
                    in self.latent_reductions[model_component]
                ], dim=0)
        return latent_code

    @staticmethod
    def to_batch_size_num_charts_contiguous(input):                             # (num_charts, batch.size, ...)
        # return a copy of the input tensor that is contiguous in the shape
        # (batch.size, num_charts, ...)
        input = input.transpose(0, 1).clone(                                    # (batch.size, num_charts, ...)  
            memory_format=torch.contiguous_format
        )
        input = input.transpose(0, 1)                                           # (num_charts, batch.size, ...)
        return input 

    def infer_signed_dist(self, latent_code_cond_sdf, mapped_pcl):
        # infer the signed distances of the mapped point cloud, for each chart
        return torch.stack([                                                    # (num_charts, [batch.size,] P)
            chart_cond_sdf(chart_cond_sdf_latent_code,                          # ([batch.size,] P)
                           chart_mapped_pcl)
            for chart_cond_sdf,
                chart_cond_sdf_latent_code,                                     # ([batch.size,] latent_dims / cond_sdf.reduced_latent_dims)
                chart_mapped_pcl                                                # ([batch.size,] P, 3)
            in zip(self.cond_sdfs, latent_code_cond_sdf, mapped_pcl)            # NA, (num_charts, [batch.size,] latent_dims / cond_sdf.reduced_latent_dims), (num_charts, [batch.size,] P, 3)
        ], dim=0)

    def estimate_label_freq(self, signed_dist):                                 # (num_charts, [batch.size,] P)
        # reshape the signed distances
        signed_dist = signed_dist.transpose(0, -2)                              # ([batch.size,] num_charts, P)
        signed_dist = signed_dist.flatten(start_dim=-2)                         # ([batch.size,], num_charts * P)
        
        # sort the signed distances in descending order & extract the assumed
        # interior signed distances
        uv_sample_size = signed_dist.shape[-1]                                  # ie. num_charts * P
        min_interior_uv_sample_size = math.ceil(
            uv_sample_size * self.hparams.min_interior_ratio
        )
        interior_signed_dist = signed_dist.sort(dim=-1, descending=True).values # ([batch.size,], uv_sample_size)
        interior_signed_dist = (                                                # ([batch.size,], min_interior_uv_sample_size)
            interior_signed_dist[..., :min_interior_uv_sample_size]
        )

        # derive the label frequency from the assumed interior signed distances
        interior_prob_occ = self.loss.signed_dist_to_prob_occ(                  # ([batch.size], min_interior_uv_sample_size)
            interior_signed_dist
        )
        label_freq = interior_prob_occ.median(dim=-1).values                    # ([batch.size])

        return label_freq

    def derive_is_occupied(self, signed_dist, label_freq):                      # (num_charts, [batch.size,] P), ([batch.size])
        # derive the signed distance threshold & binary occupancy
        signed_dist_threshold = self.loss.prob_occ_to_signed_dist(              # ([batch.size])
            label_freq * self.hparams.prob_occ_threshold
        )
        signed_dist_threshold = signed_dist_threshold.unsqueeze(dim=-1)         # ([batch.size,] 1)
        is_occupied = signed_dist > signed_dist_threshold                       # (num_charts, [batch.size,] P)
        return is_occupied

    @staticmethod
    def estimate_occupancy_ratio(is_occupied):                                  # (num_charts, [batch.size,] P)
        is_occupied = is_occupied.to(torch.get_default_dtype())                 # (num_charts, [batch.size,] P)
        occupancy_ratio = is_occupied.mean(dim=(0, -1))                         # ([batch.size])
        return occupancy_ratio

    def derive_chart_uv_sample_size(self, occupancy_ratio, stage):
        chart_uv_sample_size =  torch.ceil(                                     # ([batch.size])
            self.chart_target_pcl_nml_size[stage] / occupancy_ratio
        )
        chart_uv_sample_size = chart_uv_sample_size.clamp(
            min=None, max=self.chart_uv_max_sample_size[stage]
        ).to(torch.int64)
        return chart_uv_sample_size

    def infer_pcl(
        self,
        latent_code_cond_homeomorphism,
        input_pcl,
        reverse=False
    ):
        return torch.stack([                                                    # (num_charts, [batch.size,] P, 3)
            chart_cond_homeomorphism(chart_cond_homeomorphism_latent_code,      # ([batch.size,] P, 3)
                                     chart_input_pcl)
            if not reverse else     # support forward cond. hom. only
            chart_cond_homeomorphism(chart_cond_homeomorphism_latent_code,      # ([batch.size,] P, 3)
                                     chart_input_pcl, reverse=True)
            for chart_cond_homeomorphism,
                chart_cond_homeomorphism_latent_code,                           # ([batch.size,] latent_dims / cond_homeomorphism.reduced_latent_dims)
                chart_input_pcl                                                 # ([batch.size,] P, 3)
            in zip(self.cond_homeomorphisms,
                   latent_code_cond_homeomorphism,                              # (num_charts, [batch.size,] latent_dims / cond_homeomorphism.reduced_latent_dims)
                   input_pcl)                                                   # (num_charts, [batch.size,] P, 3)
        ], dim=0)

    def infer_mapped_tangent_vectors(
        self,
        true_padded_input_uv,                                                   # (num_charts, [batch.size,] batch.chart_uv_sample_size, 3)
        mapped_pcl,                                                             # (num_charts, [batch.size,] batch.chart_uv_sample_size, 3)
        stage
    ):
        # create the computational graph for the tangent vectors only in train
        create_graph = {
            "train": True,
            "val": False,
            "test": False
        }[stage]

        # `jacobian[*indices, :, :]` is the 3x3 transposed Jacobian matrix of
        # the cond homeomorphism with UVW-coordinate inputs & XYZ-coordinate 
        # outputs at `true_padded_input_uv[*indices, :]`
        jacobian = autograd.jacobian(                                           # (num_charts, [batch.size,] batch.chart_uv_sample_size, 3, 3)
            output=mapped_pcl,
            inputs=true_padded_input_uv,
            create_graph=create_graph
        )

        # 1. `mapped_tangent_vectors[*indices, 0, :]` is the gradient of XYZ 
        #    wrt. U & `mapped_tangent_vectors[*indices, 1, :]` is the gradient
        #    of XYZ wrt. V, at `padded_input_uv[*indices, :]`
        # 2. `mapped_tangent_vectors[*indices, 0, :]` & 
        #    `mapped_tangent_vectors[*indices, 1, :]` are the tangent vectors
        #    (not normalized to unit vectors) at `mapped_pcl[*indices, :]`
        mapped_tangent_vectors = jacobian[..., :2, :]                           # (num_charts, [batch.size,] batch.chart_uv_sample_size, 2, 3)

        return mapped_tangent_vectors

    def sample_regular_uv(self, sample_chart_uv_sample_size_sqrt):
        size = ( sample_chart_uv_sample_size_sqrt, ) * 2
        regular_uv = self.regular_sampler(size, self.device)                    # (sample_chart_uv_sample_size_sqrt, sample_chart_uv_sample_size_sqrt, 2)

        # pad zeros to the regular UV samples as (last) W-coordinates
        padded_regular_uv = torch.nn.functional.pad(regular_uv, (0, 1))         # (sample_chart_uv_sample_size_sqrt, sample_chart_uv_sample_size_sqrt, 3)

        # linearize & expand the padded regular UV samples across a new
        # chart dimension
        padded_regular_uv = padded_regular_uv.view(1, -1, 3)                    # (1, sample.chart_uv_sample_size, 3)
        padded_regular_uv = padded_regular_uv.expand(                           # (num_charts, sample.chart_uv_sample_size, 3)
            self.hparams.num_charts, -1, -1
        )

        # require gradients for the padded regular UV samples in order to
        # compute gradients of the homeomorphisms
        padded_regular_uv.requires_grad_()

        return padded_regular_uv
