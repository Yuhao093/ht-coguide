"""Remap a COCO caption ground-truth file to the first N evaluated images."""
import argparse
import json


def convert_to_gt(gt_path, select_num, output_path):
    with open(gt_path, "r", encoding="utf-8") as stream:
        gt = json.load(stream)

    annotations = []
    for index, image in enumerate(gt["images"]):
        for annotation in gt["annotations"].copy():
            if annotation["image_id"] == image["id"]:
                annotation["image_id"] = index
                annotations.append(annotation)
                gt["annotations"].remove(annotation)
        image["id"] = index

    images = gt["images"][:select_num]
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump({"annotations": annotations, "images": images}, stream,
                  indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--select_num", type=int, default=1000)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    convert_to_gt(args.gt_path, args.select_num, args.output_path)
