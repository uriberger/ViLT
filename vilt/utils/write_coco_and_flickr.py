import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions, iid2split):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    return [binary, captions, name, split]


def make_arrow(root, dataset_root):
    def generate_bs(dataset_name, json_name, image_dirs):
        with open(f"{root}/{dataset_name}/karpathy/{json_name}.json", "r") as fp:
            captions = json.load(fp)

        captions = captions["images"]

        iid2captions = defaultdict(list)
        iid2split = dict()

        for cap in tqdm(captions):
            filename = cap["filename"]
            iid2split[filename] = cap["split"]
            for c in cap["sentences"]:
                iid2captions[filename].append(c["raw"])

        path_lists = [list(glob(f"{root}/{dataset_name}/{image_dir}/*.jpg")) for image_dir in image_dirs]
        paths = sum(path_lists, [])
        random.shuffle(paths)
        caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]

        if len(paths) == len(caption_paths):
            print("all images have caption annotations")
        else:
            print("not all images have caption annotations")
        print(
            len(paths), len(caption_paths), len(iid2captions),
        )

        bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]

        return bs
    
    coco_bs = generate_bs('COCO', 'dataset_coco', ['train2014', 'val2014'])
    flickr_bs = generate_bs('flickr30', 'dataset_flickr30k', 'images')
    bs = coco_bs + flickr_bs

    for split in ["train", "val", "restval", "test"]:
        batches = [b for b in bs if b[-1] == split]

        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/coco_and_flickr_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
