# lvis_json2yolo.py
# Script to create YOLO-formatted annotation files from LVIS.
# Also creates txt files containing split information.

import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import logging
from lvis import LVIS

def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance. 
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    From https://github.com/ultralytics/JSON2YOLO/blob/master/general_json2yolo.py
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all 
    segments into one.
    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...], 
            each segmentation is a list of coordinates.
    This function taken from https://github.com/ultralytics/JSON2YOLO/blob/master/general_json2yolo.py
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def save_category_info(args, cats):
    """ Saves id: synset pairs in args.category_names (as a yaml file)
    """
    logging.info("Saving category synsets to file...")
    with open(args.out_info_path, 'w') as f:
        f.write("names:\n")
        for cat in cats:
            f.write(f"  {cat['id']}: {cat['synset']}\n")


def get_split_filepaths(args, dset):
    """ Gets the filepaths used for the dataset given. Returns them as a list.
    """
    imgs = dset.load_imgs(ids=None)
    # file paths
    filepaths = [os.path.join('./coco', 'images', img["coco_url"].split('/')[-1]) for img in imgs]    # Gives e.g. ./coco/images/000000397133.jpg.
    check_filepaths_exist(args, filepaths)
    return filepaths


def check_filepaths_exist(args, filepaths):
    """ Runs through a list of filepaths (relative to the LVIS base dir) and confirms they actually exist.
    """
    for path in filepaths:
        abspath = os.path.join(args.lvis_path, path)
        assert os.path.isfile(abspath), f"Error: file {abspath} not found"


def save_list_to_txt(fname, to_save):
    """ Convenience fn to save list of strings to a txt file """
    with open(fname, 'w') as f:
        for item in to_save:
            f.write(f"{item}\n")


def lvis_anns_to_yolo(args, dset):
    """ Generate and save annotations for all images in the LVIS dataset dset. """
    imgs = dset.load_imgs(ids=None)
    for img in imgs:
        h, w = img['height'], img['width']
        ann_ids = dset.get_ann_ids(img_ids=[img["id"]])
        anns = dset.load_anns(ann_ids)
        # Each annotation is for a single object, but may contain multiple polygons
        img_filename = img['coco_url'].split('/')[-1].split('.')[0]
        out_path = os.path.join(args.out_labels_dir, f"{img_filename}.txt")
        with open(out_path, 'w') as f:
            for ann in anns:
                if len(ann['segmentation']) > 1:
                    segments = merge_multi_segment(ann['segmentation'])
                    segments = (np.concatenate(segments, axis=0) / np.array([w, h])).reshape(-1).tolist()
                else:
                    segments = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
                    segments = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                f.write(f"{ann['category_id']} {' '.join([str(x) for x in segments])}\n")


def main(args):
    # First get categories from the training dataset, and save them to a file 
    train_set = LVIS(args.train_json)
    val_set = LVIS(args.val_json)
    test_set = LVIS(args.test_json)
    #train_cats = train_set.load_cats(ids=None)

    #save_category_info(args, train_cats)

    # Get train and test filenames/URLs (they're different to the COCO splits)
    logging.info("Getting split filepaths...")
    train_filepaths = get_split_filepaths(args, train_set)
    val_filepaths = get_split_filepaths(args, val_set)
    test_filepaths = get_split_filepaths(args, test_set)

    logging.info("Writing split filepaths to disk...")
    save_list_to_txt(args.out_train_split_path, train_filepaths)
    save_list_to_txt(args.out_test_split_path, test_filepaths)
    save_list_to_txt(args.out_val_split_path, val_filepaths)

    # For each image, create a file with instance annotations.
    # Each line is an instance in the image, with the format (for polygon masks):
    # label_id, x1, y1, x2, y2, ..., xn, yn
    #logging.info("Creating YOLO-format annotations...")
    #lvis_anns_to_yolo(args, train_set)
    #lvis_anns_to_yolo(args, test_set)
    #lvis_anns_to_yolo(args, val_set)

    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Creates YOLO-formatted annotations from a copy of LVIS for use with training YOLOv5 or YOLOv7.
        Generates annotation files for each image, txt files detailing split filenames, and another txt file containing class information
        in YAML format.
        Requires LVIS api to be installed: https://github.com/lvis-dataset/lvis-api
        """)
    parser.add_argument("--lvis_path", required=True, type=str, help="""
        /path/to/LVIS dir (containing lvis and coco subdirs).
        Expects lvis dir to contain train, test, and val json files.
        Expects coco dir to contain train/test/val folders, containing the original COCO images.
        """)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)

    # Get JSON names this way to handle future version changes. Files are currently named lvis_v1_<splitname>.json for train and val.
    args.train_json = glob.glob(os.path.join(args.lvis_path, 'lvis', '*train.json'))[0]
    args.val_json = glob.glob(os.path.join(args.lvis_path, 'lvis', '*val.json'))[0]
    args.test_json = glob.glob(os.path.join(args.lvis_path, 'lvis', '*test*.json'))[0]

    # Configure output paths
    args.out_info_path = os.path.join(args.lvis_path, "category_names.yaml")
    args.out_train_split_path = os.path.join(args.lvis_path, "lvis_train_files.txt")
    args.out_test_split_path = os.path.join(args.lvis_path, "lvis_test_files.txt")
    args.out_val_split_path = os.path.join(args.lvis_path, "lvis_val_files.txt")

    args.out_labels_dir = os.path.join(args.lvis_path, 'coco', 'labels')

    main(args)
