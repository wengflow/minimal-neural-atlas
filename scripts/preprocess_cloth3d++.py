import sys
import os
import pathlib
import shutil
import argparse
import glob
import logging
import random
import tqdm
import numpy as np
import pytorch3d.io
import pytorch3d.ops

PROJECT_DIR = os.path.join(sys.path[0], '..')
OBJ_WILDCARD = "*/*/*.obj"
STAGE_DELIMITER = "_"
SAMPLE_ID_DELIMITER = "-"
SPLIT_FILENAME_FORMAT_STR = "{}.lst"

PCL_NML_SAMPLE_SIZE = 100000
ROTATION_MATRIX = np.array([[  0, -1,  0 ],
                            [  0,  0,  1 ],
                            [ -1,  0,  0 ]])
MESH_FILENAME = "mesh.obj"
ORIENTED_POINT_CLOUD_FILENAME = "pointcloud.npz"
POINT_CLOUD_KEY = "points"
SURFACE_NORMAL_KEY = "normals"


def main(args):
    # suppress `pytorch3d.io.load_objs_as_meshes()` logger error messages
    logging.basicConfig(level=logging.CRITICAL)

    # get the filepaths of all mesh samples in the dataset
    mesh_filepaths = glob.glob(os.path.join(args.src_path, OBJ_WILDCARD))

    # randomly permute the filepaths
    random.shuffle(mesh_filepaths)

    # iteratively preprocess the meshes
    print("Processing dataset")
    dataset_splits = {}
    for mesh_filepath in tqdm.tqdm(mesh_filepaths):
        # deduce the class, seq ID & stage from the filepath of the mesh sample
        class_name = os.path.splitext(
            os.path.basename(mesh_filepath)
        )[0].lower()
        sequence_id = os.path.basename(os.path.dirname(mesh_filepath))
        stage = os.path.basename(
            os.path.dirname(os.path.dirname(mesh_filepath))
        ).split(STAGE_DELIMITER)[0]

        # deduce the unique sample ID of this mesh sample
        sample_id = SAMPLE_ID_DELIMITER.join([class_name, stage, sequence_id])

        # record this sample to the appropriate dataset split
        if not class_name in dataset_splits:
            dataset_splits[class_name] = {}
        if not stage in dataset_splits[class_name]:
            dataset_splits[class_name][stage] = []
        dataset_splits[class_name][stage].append(sample_id)

        # deduce the destination path of this sample & skip this sample, if it
        # has been processed
        sample_dst_path = os.path.join(args.dst_path, class_name, sample_id)
        if os.path.isdir(sample_dst_path):
            continue

        # copy the mesh to the sample destination path & rename it
        pathlib.Path(sample_dst_path).mkdir(parents=True)
        shutil.copy2(
            mesh_filepath, os.path.join(sample_dst_path, MESH_FILENAME)
        )

        # sample points & surface normals of this mesh
        mesh = pytorch3d.io.load_objs_as_meshes(
            [ mesh_filepath ], load_textures=False
        )
        pcl, nml = pytorch3d.ops.sample_points_from_meshes(                     # (1, PCL_NML_SAMPLE_SIZE, 3), (1, PCL_NML_SAMPLE_SIZE, 3)
            mesh, num_samples=PCL_NML_SAMPLE_SIZE, return_normals=True
        )
        pcl = pcl.squeeze(dim=0).numpy()                                        # (PCL_NML_SAMPLE_SIZE, 3)
        nml = nml.squeeze(dim=0).numpy()                                        # (PCL_NML_SAMPLE_SIZE, 3)

        # rotate this mesh such that the y & x-axes are the up & front
        # directions respectively (consistent with `ShapeNetCore_v1`)
        pcl = pcl @ ROTATION_MATRIX.T                                           # (PCL_NML_SAMPLE_SIZE, 3)
        nml = nml @ ROTATION_MATRIX.T                                           # (PCL_NML_SAMPLE_SIZE, 3)

        # save the point cloud & surface normals to the sample dest. path
        np.savez(
            os.path.join(sample_dst_path, ORIENTED_POINT_CLOUD_FILENAME),
            **{ POINT_CLOUD_KEY: pcl, SURFACE_NORMAL_KEY: nml }
        )
    print("Done")

    # save the dataset splits
    print("Saving dataset splits")
    for class_name, class_splits in tqdm.tqdm(dataset_splits.items()):
        for stage, stage_split in class_splits.items():
            split_filepath = os.path.join(
                args.dst_path, class_name,
                SPLIT_FILENAME_FORMAT_STR.format(stage)
            )
            with open(split_filepath, "w") as f:
                f.write("\n".join(stage_split))
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Script for pre-processing the raw CLOTH3D++ dataset to"
                     " generate a derivative dataset of point clouds and"
                     " surface normals")
    )
    parser.add_argument(
        "src_path", type=str, help="Path to the raw dataset."
    )
    parser.add_argument(
        "dst_path", type=str, help="Desired path of the pre-processed dataset."
    )
    args = parser.parse_args()

    main(args)
