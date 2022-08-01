import itertools
import easydict
import torch
import torchvision
import pytorch_lightning as pl
from ..utils import transforms
from . import cloth3d, shapenet, samplers


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        seed,
        input,
        target,
        num_nodes,
        gpus,
        dataset,
        dataset_directory,
        classes,
        train_dataset_ratio,
        val_dataset_ratio,
        test_dataset_ratio,
        overfit,
        train_dataset_perm_seed,
        eval_dataset_perm_seed,
        uv_space_scale,
        uv_sampling_dist,
        train_uv_max_sample_size,
        eval_uv_max_sample_size,
        pcl_nml_replace_samples,
        train_target_pcl_nml_size,
        eval_target_pcl_nml_size,
        pcl_normalization_mode,
        pcl_normalization_scale,
        train_eff_batch_size,
        val_eff_batch_size,
        test_eff_batch_size,
        num_workers_per_node,
        input_pcl=None,
        input_nml=None,
        input_img=None
    ):
        super().__init__()
        assert isinstance(overfit, bool)
        for stage_dataset_ratio in [
            train_dataset_ratio, val_dataset_ratio, test_dataset_ratio
        ]:
            assert isinstance(stage_dataset_ratio, int) \
                   or (isinstance(stage_dataset_ratio, float)
                       and 0.0 < stage_dataset_ratio <= 1.0)

        # save some non-hyperparameter configs as attributes
        self.input = input
        self.target = target
        self.dataset = dataset
        self.dataset_directory = dataset_directory
        self.classes = classes
        self.train_dataset_ratio = train_dataset_ratio
        self.val_dataset_ratio = val_dataset_ratio
        self.test_dataset_ratio = test_dataset_ratio
        self.overfit = overfit
        self.eval_dataset_perm_seed = eval_dataset_perm_seed
        self.eval_uv_max_sample_size = eval_uv_max_sample_size
        self.eval_target_pcl_nml_size = eval_target_pcl_nml_size
        self.val_eff_batch_size = val_eff_batch_size
        self.test_eff_batch_size = test_eff_batch_size

        if train_dataset_perm_seed is None:
            self.train_dataset_perm_seed = seed
        else:
            self.train_dataset_perm_seed = train_dataset_perm_seed

        if "pcl" in input:
            self.input_pcl_target_noise = input_pcl.pop("target_noise")
            self.input_pcl_trim_lower_half = input_pcl.pop(
                "trim_lower_half"
            )
        if "nml" in input:
            self.input_nml_target_noise = input_nml.pop("target_noise")

        # collate the relevant input-dependent hyperparameters
        input_to_hparam_name = {
            "pcl": "input_pcl",
            "nml": "input_nml",
            "img": "input_img"
        }
        input_hparam_names = map(input_to_hparam_name.__getitem__, input)

        # log, checkpoint & save hyperparameters to `hparams` attribute
        self.save_hyperparameters(
            "seed",
            "uv_space_scale",
            "uv_sampling_dist",
            "train_uv_max_sample_size",
            "pcl_nml_replace_samples",
            "train_target_pcl_nml_size",
            "pcl_normalization_mode",
            "pcl_normalization_scale",
            "train_eff_batch_size",
            *input_hparam_names
        )

        if gpus is None:    # ie. running on CPU
            self.train_batch_size = train_eff_batch_size
            self.val_batch_size = val_eff_batch_size
            self.test_batch_size = test_eff_batch_size
            self.num_workers = num_workers_per_node
        else:               # ie. running on GPU
            # derive gpu numbers
            num_gpus_per_node = len(gpus)
            num_gpus = num_nodes * num_gpus_per_node

            # derive the batch size per GPU across all nodes
            self.train_batch_size = train_eff_batch_size // num_gpus
            self.val_batch_size = val_eff_batch_size // num_gpus
            self.test_batch_size = test_eff_batch_size // num_gpus

            # derive the no. of workers for each subprocess associated to a GPU
            self.num_workers = num_workers_per_node // num_gpus_per_node

        # instantiate transforms
        self._init_transforms()

    def _init_transforms(self):
        """
        NOTE:
            1. For each data modality, we assume that the input and target
               data, that the input and target transforms operate on
               repectively, are identical (if both available).
            2. Target point cloud, class ID & sample ID are always available
               during training & evaluation
            3. Target surface normal is always available during evaluation
        """

        # initialize the lists of transforms to be applied successively
        """
        NOTE:
            Transforms must be applied in the `Dataset`(s) according to the
            following (insertion) order (ie. from `pcl` to `img`, from 
            `input` to `target`), which is preserved by `dict`/`EasyDict`
            from Python 3.7, since some transforms are persistent across
            several data modalities & roles
        """
        self.transforms = easydict.EasyDict({
            "train": {
                "pcl": { "input": [], "target": [] },
                "nml": { "input": [], "target": [] },
                "img": { "input": [] },
                "class_id": { "common": [] },
                "sample_id": { "common": [] }
            },
            "eval": {
                "pcl": { "input": [], "target": [] },
                "nml": { "input": [], "target": [] },
                "img": { "input": [] },
                "class_id": { "common": [] },
                "sample_id": { "common": [] }
            }
        })

        # `self.transforms.train/eval.pcl.target`, `self.transforms.train/eval
        # .class_id.common`, `self.transforms.train/eval.sample_id.common` and
        # `self.transforms.eval.nml.target` are always available
        if "pcl" not in self.input:
            self.transforms.train.pcl.pop("input")
            self.transforms.eval.pcl.pop("input")
        if "nml" not in [ *self.input, *self.target ]:
            self.transforms.train.pop("nml")
            self.transforms.eval.nml.pop("input")
        elif "nml" not in self.input:
            self.transforms.train.nml.pop("input")
            self.transforms.eval.nml.pop("input")
        elif "nml" not in self.target:
            self.transforms.train.nml.pop("target")
        if "img" not in self.input:
            self.transforms.train.pop("img")
            self.transforms.eval.pop("img")

        # instantiate joint point cloud & surface normal transforms
        if "pcl" in self.input:
            # [train, eval] trim lower half (along the y-axis)
            # trimming is performed first in order to prevent non-fixed point 
            # cloud & surface normal size, when done after subsampling
            if self.input_pcl_trim_lower_half:
                unit_ball_pcl_normalization = transforms.PclNormalization(
                    mode="ball", scale=1.
                )
                lower_y_trim = transforms.Trim(
                    trim_vector_dim=1, trim_negative=True
                )

                # process point cloud
                for stage, role in itertools.product(
                    [ "train", "eval" ], [ "input", "target" ]
                ):
                    self.transforms[stage].pcl[role].extend([
                        unit_ball_pcl_normalization.apply,
                        lower_y_trim.apply,
                        unit_ball_pcl_normalization.apply_same_inverse
                    ])

                # process surface normals
                if "nml" in self.input:
                    for stage in [ "train", "eval" ]:
                        self.transforms[stage].nml.input.append(
                            lower_y_trim.apply_same
                        )
                if "nml" in self.target:
                    for stage in [ "train", "eval" ]:
                        self.transforms[stage].nml.target.append(
                            lower_y_trim.apply_same
                        )

        # [train, eval] subsampling (done asap to reduce computations)
        random_subsampling = easydict.EasyDict({})
        if "pcl" not in self.input:
            random_subsampling.train = transforms.RandomSubsampling(
                len=self.hparams.train_target_pcl_nml_size,
                axis=0,
                replacement=self.hparams.pcl_nml_replace_samples
            )
            random_subsampling.eval = transforms.RandomSubsampling(
                len=self.eval_target_pcl_nml_size,
                axis=0,
                replacement=self.hparams.pcl_nml_replace_samples
            )

            # subsample point cloud
            for stage in [ "train", "eval" ]:
                self.transforms[stage].pcl.target.append(
                    random_subsampling[stage].apply
                )

            # subsample surface normal
            if "nml" in self.target:
                self.transforms.train.nml.target.append(
                    random_subsampling.train.apply_same
                )
            self.transforms.eval.nml.target.append(
                random_subsampling.eval.apply_same
            )            
        else:   # elif "pcl" in self.input:
            # randomly subsample input & target data such that the subsampled 
            # role data of smaller size is a subset of the larger, and
            # furthermore larger[:len(smaller), ...] = smaller
            random_subsampling.train = transforms.RandomSubsampling(
                len=max(self.hparams.input_pcl.input_pcl_nml_size,
                        self.hparams.train_target_pcl_nml_size),
                axis=0,
                replacement=self.hparams.pcl_nml_replace_samples
            )
            random_subsampling.eval = transforms.RandomSubsampling(
                len=max(self.hparams.input_pcl.input_pcl_nml_size,
                        self.eval_target_pcl_nml_size),
                axis=0,
                replacement=self.hparams.pcl_nml_replace_samples
            )

            input_seq_subsampling = transforms.SequentialSubsampling(
                len=self.hparams.input_pcl.input_pcl_nml_size, axis=0
            )
            target_seq_subsampling = easydict.EasyDict({})
            target_seq_subsampling.train = transforms.SequentialSubsampling(
                len=self.hparams.train_target_pcl_nml_size, axis=0
            )
            target_seq_subsampling.eval = transforms.SequentialSubsampling(
                len=self.eval_target_pcl_nml_size, axis=0
            )

            # subsample point cloud
            for stage in [ "train", "eval" ]:
                self.transforms[stage].pcl.input.extend([
                    random_subsampling[stage].apply,
                    input_seq_subsampling.apply
                ])
                self.transforms[stage].pcl.target.extend([
                    random_subsampling[stage].apply_same,
                    target_seq_subsampling[stage].apply
                ])

            # subsample surface normals
            if "nml" in self.input:
                for stage in [ "train", "eval" ]:
                    self.transforms[stage].nml.input.extend([
                        random_subsampling[stage].apply_same,
                        input_seq_subsampling.apply
                    ])
            if "nml" in self.target:
                self.transforms.train.nml.target.extend([
                    random_subsampling.train.apply_same,
                    target_seq_subsampling.train.apply
                ])
            self.transforms.eval.nml.target.extend([
                random_subsampling.eval.apply_same,
                target_seq_subsampling.eval.apply
            ])

        if "pcl" in self.input:
            # [train] random rotation
            if self.hparams.input_pcl.random_rotation:
                random_3d_rotation = transforms.Random3DRotation()

                # rotate point cloud
                self.transforms.train.pcl.input.append(
                    random_3d_rotation.apply
                )
                self.transforms.train.pcl.target.append(
                    random_3d_rotation.apply_same
                )

                # rotate surface normals
                if "nml" in self.input:
                    self.transforms.train.nml.input.append(
                        random_3d_rotation.apply_same
                    )
                if "nml" in self.target:
                    self.transforms.train.nml.target.append(
                        random_3d_rotation.apply_same
                    )
            
            # [train] random xy reflection
            if self.hparams.input_pcl.random_xy_reflection:
                random_xy_reflection = transforms.RandomReflection(
                    reflection_vector_dim=2, input_vector_dims=3
                )

                # reflect point cloud
                self.transforms.train.pcl.input.append(
                    random_xy_reflection.apply
                )
                self.transforms.train.pcl.target.append(
                    random_xy_reflection.apply_same
                )

                # reflect surface normals
                if "nml" in self.input:
                    self.transforms.train.nml.input.append(
                        random_xy_reflection.apply_same
                    )
                if "nml" in self.target:
                    self.transforms.train.nml.target.append(
                        random_xy_reflection.apply_same
                    )

        # instantiate joint point cloud and image transforms
        if "img" in self.input:
            # [train, eval] TODO: target_pcl_frame
            assert self.hparams.input_img.target_pcl_frame in [ 
                "object", "viewpt_object"
            ]
            if self.hparams.input_img.target_pcl_frame == "viewpt_object":
                raise NotImplementedError   # TODO

            # [train] random horizontal flip
            if self.hparams.input_img.random_hflip:
                raise NotImplementedError   # TODO

            # [train] random vertical flip
            if self.hparams.input_img.random_hflip:
                raise NotImplementedError   # TODO            


        # instantiate point cloud-only transforms
        if "pcl" in self.input:
            # [train, eval] add gaussian noise
            if self.hparams.input_pcl.input_noise_std > 0.:
                gaussian_noise = transforms.GaussianNoise(
                    mean=0., std=self.hparams.input_pcl.input_noise_std
                )

                self.transforms.train.pcl.input.append(
                    gaussian_noise.apply
                )
                if self.input_pcl_target_noise: # as augmentation
                    self.transforms.train.pcl.target.append(
                        gaussian_noise.apply_same
                    )
                else:   # as noisy/corrupted input
                    self.transforms.eval.pcl.input.append(
                        gaussian_noise.apply
                    )

        # [train, eval] normalize
        pcl_normalization = transforms.PclNormalization(
            mode=self.hparams.pcl_normalization_mode,
            scale=self.hparams.pcl_normalization_scale
        )
        if "pcl" in self.input:
            for stage in [ "train", "eval" ]:
                self.transforms[stage].pcl.input.append(
                    pcl_normalization.apply
                )
        for stage in [ "train", "eval" ]:
            self.transforms[stage].pcl.target.append(
                pcl_normalization.apply
            )

        # instantiate surface normal-only transforms
        if "nml" in self.input:
            # [train, eval] add surface normal noise
            if self.hparams.input_nml.input_noise_std > 0.:
                dirvec_noise = transforms.DirectionVectorNoise(
                    mean=0., std=self.hparams.input_nml.input_noise_std
                )

                self.transforms.train.nml.input.append(
                    dirvec_noise.apply
                )
                if self.input_nml_target_noise: # as augmentation
                    self.transforms.train.nml.target.append(
                        dirvec_noise.apply_same
                    )
                else:   # as noisy/corrupted input
                    self.transforms.eval.nml.input.append(
                        dirvec_noise.apply
                    )            

        # instantiate image-only transforms
        if "img" in self.input:
            # [train, eval] resize
            resize = torchvision.transforms.Resize(
                self.hparams.input_img.img_size, antialias=True
            )
            for stage in ["train", "eval"]:
                self.transforms[stage].img.input.append(resize)
        
            # [train] color jitter
            jitter_hparams = [
                self.hparams.input_img.brightness_jitter_factor,
                self.hparams.input_img.contrast_jitter_factor,
                self.hparams.input_img.saturation_jitter_factor,
                self.hparams.input_img.hue_jitter_factor
            ]
            if torch.any(torch.Tensor(jitter_hparams) != 0.):
                color_jitter = torchvision.transforms.ColorJitter(
                    *jitter_hparams
                )
                self.transforms.train.img.input.append(color_jitter)

        # compile the lists of transforms
        for stage_transforms in self.transforms.values():
            for modality_transforms in stage_transforms.values():
                for role in modality_transforms.keys():
                    modality_transforms[role] = torchvision.transforms.Compose(
                        modality_transforms[role]
                    )

    def setup(self, stage):
        # instantiate datasets with transforms & uv samplers
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset("train")
            if self.overfit:
                self.val_dataset = self.train_dataset
            else:
                self.val_dataset = self._build_dataset("val")
            self.train_uv_sampler = self._build_uv_sampler("train")
            self.val_uv_sampler = self._build_uv_sampler("val")
        if stage in (None, "validate"):
            if self.overfit:
                self.val_dataset = getattr(
                    self, "train_dataset", self._build_dataset("train")
                )
            else:
                self.val_dataset = self._build_dataset("val")
            self.val_uv_sampler = self._build_uv_sampler("val")
        if stage in (None, "test"):
            if self.overfit:
                self.test_dataset = getattr(
                    self, "train_dataset", self._build_dataset("train")
                )
            else:
                self.test_dataset = self._build_dataset("test")
            self.test_uv_sampler = self._build_uv_sampler("test")

    def _build_dataset(self, stage):
        DatasetCls = {
            "cloth3d++": cloth3d.Cloth3d,
            "shapenet": shapenet.ShapeNet
        }[self.dataset]

        """
        NOTE:
            1. Point cloud, class ID & sample ID are always available during
               training and evaluation
            2. Surface normal is always available during evaluation
        """
        modalities = set(
            [ "pcl", *self.input, *self.target, "class_id", "sample_id" ]
        )
        if stage == "train":
            transforms_stage = "train"
            permutation_seed = self.train_dataset_perm_seed
        else:   # elif stage in [ "val", "test" ]
            transforms_stage = "eval"
            modalities.add("nml")
            permutation_seed = self.eval_dataset_perm_seed

        dataset = DatasetCls(
            self.dataset_directory,
            modalities,
            self.transforms[transforms_stage],
            stage,
            permutation_seed,
            self.classes
        )

        # extract & return a subset of the dataset
        stage_dataset_ratio = {
            "train": self.train_dataset_ratio,
            "val": self.val_dataset_ratio,
            "test": self.test_dataset_ratio
        }[stage]
        if isinstance(stage_dataset_ratio, int):
            stage_eff_batch_size = {
                "train": self.hparams.train_eff_batch_size,
                "val": self.val_eff_batch_size,
                "test": self.test_eff_batch_size
            }[stage]
            dataset_subset_len = stage_dataset_ratio * stage_eff_batch_size
            assert dataset_subset_len <= len(dataset)
        else:   # elif isinstance(self.dataset_ratio, float)
            dataset_subset_len = int(stage_dataset_ratio * len(dataset))
        dataset_subset_indices = torch.arange(dataset_subset_len)
        dataset_subset = torch.utils.data.Subset(
            dataset, dataset_subset_indices
        )
        return dataset_subset

    def _build_uv_sampler(self, stage):
        """
        instantiate a UV sampler, which is an iterable-style dataset, that
        yields unlimited batches of UV samples with values in the range of 
        `[ -self.hparams.uv_space_scale, self.hparams.uv_space_scale ]**2`
        distributed according to `self.hparams.uv_sampling_dist`, with shape
        `(stage_batch_size, stage_uv_max_sample_size, 2)`
        """
        # TODO: "poisson-disk"        
        SamplerCls = {
            "uniform": samplers.UniformSampler
        }[self.hparams.uv_sampling_dist]

        stage_batch_size = {
            "train": self.train_batch_size,
            "val": self.val_batch_size,
            "test": self.test_batch_size
        }[stage]
        stage_uv_max_sample_size = {
            "train": self.hparams.train_uv_max_sample_size,
            "val": self.eval_uv_max_sample_size,
            "test": self.eval_uv_max_sample_size
        }[stage]

        UV_SAMPLE_DIMS = 2
        return SamplerCls(
            low=-self.hparams.uv_space_scale,
            high=self.hparams.uv_space_scale,
            size=(stage_batch_size, stage_uv_max_sample_size, UV_SAMPLE_DIMS)
        )

    def train_dataloader(self):
        """
        NOTE:
            `pl.Trainer(replace_sampler_ddp=True, accelerator=ddp/ddp_spawn, 
            ...)` implicitly replaces the `sampler` of `dataset_dataloader`
            with `DistributedSampler(shuffle=True, drop_last=False, ...)`
        """
        dataset_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,            
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        uv_sampler_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_uv_sampler,
            batch_size=1,
            num_workers=1,  # sufficient because no IO operations are involved
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )

        # does not return an `easydict` due to its lack of support of
        # initialization with a list of key-value tuples, which causes issues
        # when using `DistributedDataParallel`
        return {
            "dataset": dataset_dataloader, "input_uv": uv_sampler_dataloader
        }

    def val_dataloader(self):
        """
        NOTE:
            1. `pl.Trainer(replace_sampler_ddp=True, accelerator=ddp/ddp_spawn, 
               ...)` implicitly replaces the `sampler` of `dataset_dataloader`
               with `DistributedSampler(shuffle=True, drop_last=False, ...)`.
            2. If `len(self.val_dataset)` is not divisible by `num_replicas`,
               validation is not entirely accurate, irrespective of
               `drop_last`.
        """
        dataset_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,            
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )
        uv_sampler_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_uv_sampler,
            batch_size=1,
            num_workers=1,  # sufficient because no IO operations are involved
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )

        # `CombinedLoader` is required to load multiple datasets simultaneously
        # during validation
        return pl.trainer.supporters.CombinedLoader(
            # does not return an `easydict` due to its lack of support of
            # initialization with a list of key-value tuples, which causes
            # issues when using `DistributedDataParallel`
            loaders={
                "dataset": dataset_dataloader,
                "input_uv": uv_sampler_dataloader
            },
            mode="min_size"
        )

    def test_dataloader(self):
        """
        NOTE:
            1. `pl.Trainer(replace_sampler_ddp=True, accelerator=ddp/ddp_spawn, 
               ...)` implicitly replaces the `sampler` of `dataset_dataloader`
               with `DistributedSampler(shuffle=True, drop_last=False, ...)`.
            2. If `len(self.test_dataset)` is not divisible by `num_replicas`,
               testing is not entirely accurate, irrespective of `drop_last`.
        """
        dataset_dataloader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )
        uv_sampler_dataloader = torch.utils.data.DataLoader(
            dataset=self.test_uv_sampler,
            batch_size=1,
            num_workers=1,  # sufficient because no IO operations are involved
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )

        # `CombinedLoader` is required to load multiple datasets simultaneously
        # during testing
        return pl.trainer.supporters.CombinedLoader(
            # does not return an `easydict` due to its lack of support of
            # initialization with a list of key-value tuples, which causes
            # issues when using `DistributedDataParallel`
            loaders={
                "dataset": dataset_dataloader,
                "input_uv": uv_sampler_dataloader
            },
            mode="min_size"
        )
