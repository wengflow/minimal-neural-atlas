import os
import abc
import easydict
import torch


class MultiModalityDataset(torch.utils.data.Dataset, abc.ABC):
    """
    TODO:
        Allow for `transforms=None`, `transforms={ modality: None }`,
        `transforms={ <some modalities not specified> }` &
        `transforms={ modality: { variant: None } }`
    """
    STAGES = [ "train", "val", "test" ]

    def __init__(
        self,
        dataset_directory,
        modalities,
        transforms,
        stage,
        permutation_seed,
        *collate_path_components_args,
        **collate_path_components_kwargs
    ):
        """
        NOTE:
            Transforms are applied to the various modalities of each
            multi-modality data sample according to the insertion order of
            modality transforms in `transforms`, which is preserved by
            `dict`/`EasyDict` from Python 3.7
        """
        super().__init__()

        assert os.path.isdir(dataset_directory)
        assert set(modalities).issubset(set(self.MODALITIES))
        assert isinstance(self.MODALITY_TO_DATASET_CLS, dict)

        assert isinstance(transforms, dict)
        # assert set(transforms.keys()).issubset(set(modalities))
        assert set(transforms.keys()) == set(modalities)
        assert all(isinstance(modality_transforms, dict)
                   for modality_transforms in transforms.values())

        assert stage in self.STAGES
        assert isinstance(permutation_seed, int) or permutation_seed is None

        path_components_of_samples = self._collate_path_components_of_samples(
            dataset_directory, modalities, stage, permutation_seed,
            *collate_path_components_args, **collate_path_components_kwargs
        )
        # `self.single_modality_datasets` preserves the insertion order of 
        # modalities in `transforms`
        self.single_modality_datasets = easydict.EasyDict({
            modality: self.MODALITY_TO_DATASET_CLS[modality](
                          transforms[modality], *path_components_of_samples
                      )
            for modality in transforms.keys()
        })
        assert all(isinstance(dataset, SingleModalityDataset)
                   for dataset in self.single_modality_datasets.values())

        self.variants = {
            variant_name
            for modality_transforms in transforms.values()
            for variant_name in modality_transforms.keys()
        }
    
    # NOTE: class properties are only supported in Python 3.9 & above,
    #       currently workaround with instance properties instead
    # @classmethod
    @property
    @abc.abstractmethod
    def MODALITY_TO_DATASET_CLS(self):
        pass

    # NOTE: class properties are only supported in Python 3.9 & above,
    #       currently workaround with instance properties instead
    # @classmethod
    @property
    def MODALITIES(self):
        return list(self.MODALITY_TO_DATASET_CLS.keys())

    @classmethod
    @abc.abstractmethod
    def _collate_path_components_of_samples(
        cls,
        dataset_directory,
        modalities,
        stage,
        permutation_seed,
        *args,
        **kwargs
    ):
        pass

    def __getitem__(self, index):
        """
        Returns:
            variant_modality_sample_dict (dict[str, dict[str, Any]]):
                Variants of the multi-modality data sample with ID `index`,
                structured according to `transforms`, but `variant_name`
                is indexed before `modality` (ie. "reverse" of `transforms`).
        """
        variant_modality_sample_dict = easydict.EasyDict({
            variant_name: {} for variant_name in self.variants
        })
        # transformed single-modality data samples are retrieved in the 
        # insertion order of modality transforms in `transforms`
        for modality, dataset in self.single_modality_datasets.items():
            for variant_name, sample_variant in dataset[index].items():
                variant_modality_sample_dict[variant_name][modality] = (
                    sample_variant
                )
        return variant_modality_sample_dict

    def __len__(self):
        first_dataset = next(iter(self.single_modality_datasets.values()))
        return len(first_dataset)


class SingleModalityDataset(torch.utils.data.Dataset, abc.ABC):
    """
    TODO:
        Allow for `transforms=None` & `transforms={ variant: None }`
    """

    def __init__(self, transforms, *path_components_of_samples):
        """
        Args:
            transforms (dict[str, Callable]):
                Mapping from the name of single modality data variant to the
                transformation function used to generate the variant from the
                raw single-modality data.
            path_components_of_samples (List[List[str]]):
                Eg. [ [ "/dataset" , "/dataset", "/dataset" , "/dataset" ]
                      [ "car"      , "car"     , "bike"     , "bike"     ],
                      [ "001"      , "002"     , "003"      , "004"      ] ]
                where it encodes 3 directory path components for each of the 
                4 data samples, giving the 4 sample directory paths 
                "/dataset/car/001", "/dataset/car/002", "/dataset/bike/003" &
                "/dataset/bike/004"
        NOTE:
            1. Transforms are applied to the variants of each single-modality
               data sample according to the insertion order of variant
               transforms in `transforms`, which is preserved by `dict`
               /`EasyDict` from Python 3.7
            2. This form of path encoding is more memory-efficient because
               strings that correspond to the common dataset directory (eg.
               "/dataset") and common class folder name (eg. "car" & "bike")
               can be shared across samples by referencing the same string.
               Moreover, sample directory paths and hence samples can also be
               easily indexed in a linear fashion.
        """
        super().__init__()

        num_path_components = len(path_components_of_samples)
        assert num_path_components > 0

        first_component_num_samples = len(path_components_of_samples[0])
        for path_component_of_samples in path_components_of_samples:
            component_num_samples = len(path_component_of_samples)
            assert component_num_samples == first_component_num_samples
            # comment the assertion below when debugging to speed up execution
            # assert all(isinstance(path_component, str)
            #            for path_component in path_component_of_samples)
        assert isinstance(transforms, dict)
        assert all(isinstance(variant_name, str)
                   for variant_name in transforms.keys())
        assert all(callable(variant_transforms)
                   for variant_transforms in transforms.values())

        self.path_components_of_samples = path_components_of_samples
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Returns:
            sample_variants (dict[str, Any]): Variants of the single-modality 
                                              data sample with ID `index`,
                                              structured according to 
                                              `self.transforms`.
        """
        # assemble the directory path of the data sample with ID `index`
        num_path_components = len(self.path_components_of_samples)
        sample_path_components = [ None ] * num_path_components
        for component_index in range(num_path_components):
            sample_path_components[component_index] = (
                self.path_components_of_samples[component_index][index]
            )
        sample_directory = os.path.join(*sample_path_components)

        # retrieve and transform the sample, according to the insertion order
        # of variant transforms in `self.transforms`, to generate variants
        sample = self._getitem_from_dir(sample_directory)
        sample_variants = easydict.EasyDict({
            variant_name: variant_transform(sample)
            for variant_name, variant_transform in self.transforms.items()
        })
        return sample_variants

    @classmethod
    @abc.abstractmethod
    def _getitem_from_dir(cls, directory):
        pass

    def __len__(self):
        return len(self.path_components_of_samples[0])
