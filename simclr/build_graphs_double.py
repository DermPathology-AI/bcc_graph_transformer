import argparse
import glob
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models


def save_coords(txt_file, csv_file_path):
    for path in csv_file_path:
        x, y = os.path.basename(path).split(".")[0].split("_")
        txt_file.writelines(f"{x}\t{y}\n")
    txt_file.close()


def adj_matrix(csv_file_path):
    total = len(csv_file_path)
    adj_s = np.zeros((total, total), dtype=np.uint8)

    coords = [
        tuple(map(int, os.path.basename(p).split(".")[0].split("_")))
        for p in csv_file_path
    ]

    for i in range(total - 1):
        xi, yi = coords[i]
        for j in range(i + 1, total):
            xj, yj = coords[j]
            # spatial (8-neighborhood)
            if abs(xi - xj) <= 1 and abs(yi - yj) <= 1:
                adj_s[i, j] = 1
                adj_s[j, i] = 1

    return torch.from_numpy(adj_s)


def compute_feats(
    args,
    bags_list,
    i_classifier,
    save_path=None,
    holdout_set=False,
    separate_feat=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_bags = len(bags_list)

    for i in range(num_bags):
        if args.magnification == "20x" and separate_feat == "high":
            csv_file_path = glob.glob(os.path.join(bags_list[i], "*", "*.jpeg"))
        else:
            # covers 'low' and default 20x (flat)
            csv_file_path = glob.glob(os.path.join(bags_list[i], "*.jpeg"))

        file_name = bags_list[i].rstrip("/").split("/")[-1]
        print(f"{len(csv_file_path)} files to be processed: {file_name}")

        if separate_feat == "low":
            simclr_files_folder = "simclr_files_double_low"
            bcc_folder = "bcc_low"
        elif separate_feat == "high":
            simclr_files_folder = "simclr_files_double_high"
            bcc_folder = "bcc_high"
        elif separate_feat == "concat":
            simclr_files_folder = "simclr_files_double_concat"
            bcc_folder = "bcc_concat"
        else:
            simclr_files_folder = "simclr_files_double"
            bcc_folder = "bcc"

        out_dir = os.path.join(save_path, simclr_files_folder, f"{file_name}_files")
        if os.path.isdir(out_dir) or len(csv_file_path) < 1:
            print("already exists")
            continue
        os.makedirs(out_dir, exist_ok=True)

        # coords
        with open(os.path.join(out_dir, "c_idx.txt"), "w+") as txt_file:
            save_coords(txt_file, csv_file_path)

        # features: read precomputed CSVs
        base = "datasets_test_set" if holdout_set else "datasets"
        csv_path = os.path.join(
            "/workspace/data/cv_methods/gt",
            base,
            bcc_folder,
            "data",
            f"{file_name}.csv",
        )
        features_double = pd.read_csv(csv_path)
        output = torch.tensor(
            np.array(features_double), dtype=torch.float32, device=device
        )

        torch.save(output.cpu(), os.path.join(out_dir, "features.pt"))

        # adjacency
        adj_s = adj_matrix(csv_file_path)
        torch.save(adj_s, os.path.join(out_dir, "adj_s.pt"))

        print(f"\r Computed: {i + 1}/{num_bags}")

        # sanity
        adj_loaded = torch.load(os.path.join(out_dir, "adj_s.pt"), map_location="cpu")
        feats_loaded = torch.load(
            os.path.join(out_dir, "features.pt"), map_location="cpu"
        )
        assert adj_loaded.shape[0] == feats_loaded.shape[0]


def main():
    parser = argparse.ArgumentParser(
        description="Compute BCC features from SimCLR embedder (graph prep)"
    )
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_feats", default=512, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--dataset", default=None, type=str, help='e.g. "path/to/data/*/"'
    )
    parser.add_argument(
        "--backbone",
        default="resnet18",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
    )
    parser.add_argument("--magnification", default="20x", type=str)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--holdout_set", action="store_true")
    parser.add_argument(
        "--separate_feat",
        default=None,
        type=str,
        choices=[None, "low", "high", "concat"],
    )

    args = parser.parse_args()

    # backbone (not used)
    if args.backbone == "resnet18":
        resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 512
    elif args.backbone == "resnet34":
        resnet = models.resnet34(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 512
    elif args.backbone == "resnet50":
        resnet = models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 2048
    else:  # resnet101
        resnet = models.resnet101(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 2048

    for p in resnet.parameters():
        p.requires_grad = False
    resnet.fc = nn.Identity()

    # keep IClassifier construction for compatibility (even if unused here)
    import cl as cl  # local import to avoid failing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i_classifier = cl.IClassifier(resnet, num_feats, output_class=args.num_classes).to(
        device
    )

    # weights not used
    if args.weights is None:
        print(
            "No feature extractor (weights not provided). Proceeding with precomputed CSV features."
        )
    else:
        state_dict_weights = torch.load(args.weights, map_location="cpu")

        for k in [
            "module.l1.weight",
            "module.l1.bias",
            "module.l2.weight",
            "module.l2.bias",
            "l1.weight",
            "l1.bias",
            "l2.weight",
            "l2.bias",
        ]:
            state_dict_weights.pop(k, None)

        state_dict_init = i_classifier.state_dict()
        new_state_dict = OrderedDict()
        for (k, v), (k_0, _) in zip(
            state_dict_weights.items(), state_dict_init.items()
        ):
            new_state_dict[k_0] = v
        i_classifier.load_state_dict(new_state_dict, strict=False)

    os.makedirs(args.output, exist_ok=True)

    # discover bags
    if args.dataset:
        bags_list = glob.glob(args.dataset)
    else:
        if args.holdout_set:
            bags_list = glob.glob(
                "/workspace/data/cv_methods/gt/WSI_test_set/bcc/pyramid/data/*/"
            )
        else:
            bags_list = glob.glob(
                "/workspace/data/cv_methods/gt/WSI/bcc/pyramid/data/*/"
            )

    if not bags_list:
        raise FileNotFoundError(
            "No bags found. Provide --dataset with a valid glob like 'path/to/data/*/'."
        )

    compute_feats(
        args,
        bags_list,
        i_classifier,
        args.output,
        holdout_set=args.holdout_set,
        separate_feat=args.separate_feat,
    )


if __name__ == "__main__":
    main()
