import os
import itertools
import easydict
import numpy as np
import torch
import torchvision
from ..utils import dataset_mixins


class ShapeNetClassId(dataset_mixins.SingleModalityDataset):
    IMAGE_FOLDER_NAME = "img_choy2016"
    NORMALIZED_CLASS_ID_CHAR_LEN = 8

    def __init__(self, transforms, *path_components_of_samples):
        super().__init__(transforms, *path_components_of_samples)

    @classmethod
    def _getitem_from_dir(cls, directory):
        # extract the class id
        if cls.IMAGE_FOLDER_NAME in directory:
            # remove trailing ".../img_choy2016/0xx"
            directory = os.path.dirname(os.path.dirname(directory))
        # remove trailing ".../<sample id>"
        directory = os.path.dirname(directory)
        class_id = os.path.basename(directory)

        # explicitly normalize the 8-character sample id for safety
        normalized_class_id = class_id.ljust(
            cls.NORMALIZED_CLASS_ID_CHAR_LEN
        )

        # convert the 8-character class id to a tensor of Unicode code point
        # integers with shape (8)
        normalized_class_id_tensor = torch.as_tensor(
            list(map(ord, normalized_class_id))
        )
        return normalized_class_id_tensor


class ShapeNetSampleId(dataset_mixins.SingleModalityDataset):
    IMAGE_FOLDER_NAME = "img_choy2016"
    NORMALIZED_SAMPLE_ID_CHAR_LEN = 37

    def __init__(self, transforms, *path_components_of_samples):
        super().__init__(transforms, *path_components_of_samples)

    @classmethod
    def _getitem_from_dir(cls, directory):
        # extract the sample id
        if cls.IMAGE_FOLDER_NAME in directory:
            # remove trailing ".../img_choy2016/0xx"
            directory = os.path.dirname(os.path.dirname(directory))
        sample_id = os.path.basename(directory)

        # normalize the 31/32/37-character sample id to 37 characters by
        # padding it with a trailing space whenever necessary
        normalized_sample_id = sample_id.ljust(
            cls.NORMALIZED_SAMPLE_ID_CHAR_LEN
        )

        # convert the 37-character normalized sample id to a tensor of Unicode
        # code point integers with shape (37)
        normalized_sample_id_tensor = torch.as_tensor(
            list(map(ord, normalized_sample_id))
        )
        return normalized_sample_id_tensor


class ShapeNetIntrinsics(dataset_mixins.SingleModalityDataset):
    CAMERA_PARAMS_FILENAME = "cameras.npz"
    INTRINSICS_KEY_FORMAT_STR = "camera_mat_{:d}"

    def __init__(self, transforms, *path_components_of_samples):
        super().__init__(transforms, *path_components_of_samples)

    @classmethod
    def _getitem_from_dir(cls, directory):
        # extract & remove trailing image id ".../0xx"
        img_id = int(os.path.basename(directory))
        directory = os.path.dirname(directory)
        
        filepath = os.path.join(directory, cls.CAMERA_PARAMS_FILENAME)
        camera_params = np.load(filepath)
        intrinsics = camera_params[
            cls.INTRINSICS_KEY_FORMAT_STR.format(img_id)
        ]

        # loaded intrinsics `ndarray` has a data-type of `float64`
        return torch.from_numpy(intrinsics).to(torch.get_default_dtype())


class ShapeNetExtrinsics(dataset_mixins.SingleModalityDataset):
    CAMERA_PARAMS_FILENAME = "cameras.npz"
    EXTRINSICS_KEY_FORMAT_STR = "world_mat_{:d}"

    def __init__(self, transforms, *path_components_of_samples):
        super().__init__(transforms, *path_components_of_samples)

    @classmethod
    def _getitem_from_dir(cls, directory):
        # extract & remove trailing image id ".../0xx"
        img_id = int(os.path.basename(directory))
        directory = os.path.dirname(directory)
        
        filepath = os.path.join(directory, cls.CAMERA_PARAMS_FILENAME)
        camera_params = np.load(filepath)
        extrinsics = camera_params[
            cls.EXTRINSICS_KEY_FORMAT_STR.format(img_id)
        ]

        # loaded extrinsics `ndarray` has a data-type of `float64`
        return torch.from_numpy(extrinsics).to(torch.get_default_dtype())


class ShapeNetPointCloud(dataset_mixins.SingleModalityDataset):
    IMAGE_FOLDER_NAME = "img_choy2016"
    ORIENTED_POINT_CLOUD_FILENAME = "pointcloud.npz"
    POINT_CLOUD_KEY = "points"

    def __init__(self, transforms, *path_components_of_samples):
        super().__init__(transforms, *path_components_of_samples)

    @classmethod
    def _getitem_from_dir(cls, directory):
        if cls.IMAGE_FOLDER_NAME in directory:
            # remove trailing ".../img_choy2016/0xx"
            directory = os.path.dirname(os.path.dirname(directory))
        filepath = os.path.join(directory, cls.ORIENTED_POINT_CLOUD_FILENAME)
        oriented_pcl = np.load(filepath)
        pcl = oriented_pcl[cls.POINT_CLOUD_KEY]

        # loaded point cloud `ndarray` has a data-type of `float16`
        return torch.from_numpy(pcl).to(torch.get_default_dtype())

class ShapeNetSurfaceNormal(dataset_mixins.SingleModalityDataset):
    IMAGE_FOLDER_NAME = "img_choy2016"
    ORIENTED_POINT_CLOUD_FILENAME = "pointcloud.npz"
    SURFACE_NORMAL_KEY = "normals"

    def __init__(self, transforms, *path_components_of_samples):
        super().__init__(transforms, *path_components_of_samples)

    @classmethod
    def _getitem_from_dir(cls, directory):
        if cls.IMAGE_FOLDER_NAME in directory:
            # remove trailing ".../img_choy2016/0xx"
            directory = os.path.dirname(os.path.dirname(directory))        
        filepath = os.path.join(directory, cls.ORIENTED_POINT_CLOUD_FILENAME)
        oriented_pcl = np.load(filepath)
        nml = oriented_pcl[cls.SURFACE_NORMAL_KEY]

        # loaded surface normal `ndarray` has a data-type of `float16`
        return torch.from_numpy(nml).to(torch.get_default_dtype())


class ShapeNetImage(dataset_mixins.SingleModalityDataset):
    IMAGE_EXTENSION = ".jpg"
    MAX_PIXEL_VALUE = 255

    def __init__(self, transforms, *path_components_of_samples):
        super().__init__(transforms, *path_components_of_samples)

    @classmethod
    def _getitem_from_dir(cls, directory):
        filepath = directory + cls.IMAGE_EXTENSION
        # NOTE: some 3D-R2N2 image renders are grayscale
        image = torchvision.io.read_image(
            filepath, mode=torchvision.io.ImageReadMode.RGB
        )
        # normalize pixel values from [0, 255] to [0.0, 1.0]
        image = image.to(torch.get_default_dtype()) / cls.MAX_PIXEL_VALUE

        return image


class ShapeNet(dataset_mixins.MultiModalityDataset):
    # Reference: `dataset_directory`/metadata.yaml
    CLASS_TO_ID = easydict.EasyDict({
        "table": "04379243",        # 8509 samples
        "car": "02958343",          # 7496 samples
        "chair": "03001627",        # 6778 samples
        "airplane": "02691156",     # 4045 samples
        "sofa": "04256520",         # 3173 samples
        "rifle": "04090263",        # 2372 samples
        "lamp": "03636649",         # 2318 samples
        "vessel": "04530566",       # 1939 samples
        "bench": "02828884",        # 1816 samples
        "loudspeaker": "03691459",  # 1618 samples
        "cabinet": "02933112",      # 1572 samples
        "display": "03211117",      # 1095 samples
        "telephone": "04401088"     # 1052 samples
    })                      # Total: 43783 samples
    MODALITY_TO_DATASET_CLS = easydict.EasyDict({
        "class_id": ShapeNetClassId,
        "sample_id": ShapeNetSampleId,
        "intrinsics": ShapeNetIntrinsics,
        "extrinsics": ShapeNetExtrinsics,
        "pcl": ShapeNetPointCloud,
        "nml": ShapeNetSurfaceNormal,
        "img": ShapeNetImage
    })
    SPLIT_FILENAME_FORMAT_STR = "{}.lst"
    IMG_FILENAME_FORMAT_STR = "{:03d}"
    IMG_FOLDER_NAME = "img_choy2016"
    NUM_IMAGES = 24

    def __init__(
        self,
        dataset_directory,
        modalities,
        transforms,
        stage,
        permutation_seed,
        classes
    ):
        if classes == "all":
            classes = list(self.CLASS_TO_ID.keys())
        else:
            assert set(classes).issubset(set(self.CLASS_TO_ID.keys()))
        super().__init__(
            dataset_directory, modalities, transforms,
            stage, permutation_seed, classes
        )

    @classmethod
    def _collate_path_components_of_samples(
        cls,
        dataset_directory,
        modalities,
        stage,
        permutation_seed,
        classes
    ):
        split_filename = cls.SPLIT_FILENAME_FORMAT_STR.format(stage)
        class_ids = list(map(cls.CLASS_TO_ID.__getitem__, classes))
        class_id_to_sample_ids = {}
        for class_id in class_ids:
            class_split_file_path = os.path.join(
                dataset_directory, class_id, split_filename
            )
            with open(class_split_file_path, 'r') as f:
                class_id_to_sample_ids[class_id] = f.read().splitlines()

        if "img" not in modalities:
            sample_ids = list(itertools.chain.from_iterable(
                class_id_to_sample_ids.values()
            ))
            class_id_of_samples = [
                class_id
                for class_id, class_sample_ids
                    in class_id_to_sample_ids.items()
                for _ in class_sample_ids
            ]
            num_samples = len(sample_ids)
            dataset_directory_of_samples = num_samples * [ dataset_directory ]
            path_components_of_samples = [
                dataset_directory_of_samples, class_id_of_samples, sample_ids
            ]
        else:
            id_of_samples = list(itertools.chain.from_iterable(
                cls.NUM_IMAGES * [ class_sample_id ]
                for class_sample_ids in class_id_to_sample_ids.values()
                for class_sample_id in class_sample_ids
            ))
            class_id_of_samples = list(itertools.chain.from_iterable(
                len(class_sample_ids) * cls.NUM_IMAGES * [ class_id ]
                for class_id, class_sample_ids
                in class_id_to_sample_ids.items()
            ))
            num_samples = len(id_of_samples)
            dataset_directory_of_samples = num_samples * [ dataset_directory ]
            img_folder_name_of_samples = num_samples * [ cls.IMG_FOLDER_NAME ]
            img_id_of_samples = num_samples // cls.NUM_IMAGES * list(
                map(cls.IMG_FILENAME_FORMAT_STR.format, range(cls.NUM_IMAGES))
            )
            path_components_of_samples = [
                dataset_directory_of_samples, class_id_of_samples,
                id_of_samples, img_folder_name_of_samples, img_id_of_samples
            ]

        # fixed random permutation of `path_components_of_samples` not required
        if permutation_seed is None:
            return path_components_of_samples

        # randomly permutate `path_components_of_samples` according to the
        # given `permutation_seed`, independent of the default rng, to allow
        # for the use of `overfit_batches` in `pl.Trainer()`
        generator = np.random.Generator(np.random.PCG64(permutation_seed))
        perm_indices = generator.permutation(num_samples)
        for component_index, path_component_of_samples in enumerate(
            path_components_of_samples
        ):
            path_components_of_samples[component_index] = list(
                map(path_component_of_samples.__getitem__, perm_indices)
            )

        return path_components_of_samples
