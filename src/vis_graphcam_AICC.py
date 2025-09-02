import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import openslide
import argparse


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def cam_to_mask(gray, patches, cam_matrix, w, h, w_s, h_s):
    mask = np.full_like(gray, 0.0).astype(np.float32)
    for ind1, patch in enumerate(patches):
        x, y = patch.split(".")[0].split("_")
        x, y = int(x), int(y)
        if y < 5 or x > w - 5 or y > h - 5:
            continue
        try:
            mask[
                int(y * h_s) : int((y + 1) * h_s), int(x * w_s) : int((x + 1) * w_s)
            ].fill(cam_matrix[ind1][0])
        except Exception as e:
            print(e)
    return mask


def main(args):
    file_name, label = open(args.path_file, "r").readlines()[0].split("\t")
    print("file name:", file_name)
    # site, file_name = file_name.split('/')

    site = "bcc"
    file_path = os.path.join(args.path_patches, "{}_files".format(file_name))

    print(args.path_patches)
    print("File name", file_name, "Label:", label)

    p = torch.load("graphcam/prob.pt").cpu().detach().numpy()[0]
    p_pred = p.argmax()
    print("Probablity:", p, "predicted_class:", p_pred)

    # file_path = os.path.join(args.path_patches, '{}_files/20.0/'.format(file_name))
    file_path = os.path.join(args.path_patches, "{}_files/".format(file_name))

    ori = openslide.OpenSlide(os.path.join(args.path_WSI, "{}.ndpi").format(file_name))
    patch_info = open(
        os.path.join(args.path_graph, file_name + "_files", "c_idx.txt"), "r"
    )
    print(os.path.join(args.path_graph, file_name, "c_idx.txt"))

    width, height = ori.dimensions
    print("Original dimensions", width, height)

    w, h = int(width / 512), int(height / 512)
    w_r, h_r = int(width / 20), int(height / 20)
    print("Resized factors:", w, h, w_r, h_r)

    resized_img = ori.get_thumbnail((w_r, h_r))
    resized_img = resized_img.resize((w_r, h_r))

    mag_scale = 3.5
    w_s, h_s = (
        float(512 / 20) * mag_scale,
        float(512 / 20) * mag_scale,
    )
    print(w_s, h_s)

    patch_info = patch_info.readlines()
    patches = []
    xmax, ymax = 0, 0
    for patch in patch_info:
        x, y = patch.strip("\n").split("\t")
        if xmax < int(x):
            xmax = int(x)
        if ymax < int(y):
            ymax = int(y)
        patches.append("{}_{}.jpeg".format(x, y))

    output_img = np.asarray(resized_img)[:, :, ::-1].copy()
    # -----------------------------------------------------------------------------------------------------#
    # GraphCAM
    print("visulize GraphCAM")
    assign_matrix = torch.load("graphcam/s_matrix_ori.pt")
    m = nn.Softmax(dim=1)
    assign_matrix = m(assign_matrix)

    # Thresholding for better visualization
    print("Probability:", p)

    p = np.clip(p, 0.8, 1)

    vis_all = [output_img]

    num_classes = 2
    for c in range(0, num_classes):
        # Load graphcam for differnet class
        if c == p_pred:
            cam_matrix = torch.load(f"graphcam/cam_{c}.pt")
            cam_matrix = torch.mm(assign_matrix, cam_matrix.transpose(1, 0))
            cam_matrix = cam_matrix.cpu()

            # Normalize the graphcam
            cam_matrix = (cam_matrix - cam_matrix.min()) / (
                cam_matrix.max() - cam_matrix.min()
            )
            cam_matrix = cam_matrix.detach().numpy()
            cam_matrix = p[0] * cam_matrix
            cam_matrix = np.clip(cam_matrix, 0, 1)

            output_img_copy = np.copy(output_img)
            gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            image_transformer_attribution = (
                output_img_copy - output_img_copy.min()
            ) / (output_img_copy.max() - output_img_copy.min())

            mask = cam_to_mask(gray, patches, cam_matrix, w, h, w_s, h_s)

            vis = show_cam_on_image(image_transformer_attribution, mask)
            vis = np.uint8(255 * vis)
            vis_all.append(vis)

    h, w, _ = output_img.shape

    if h > w:
        vis_merge = cv2.hconcat(vis_all)
    else:
        vis_merge = cv2.vconcat(vis_all)

    cv2.imwrite("graphcam_vis/cam_all.png", np.rot90(vis_merge, axes=(1, 0)))
    cv2.imwrite(
        "graphcam_vis/{}_{}_cam_all_{}.png".format(
            file_name, site, str(round(max(p), 5))
        ),
        np.rot90(vis_merge, axes=(1, 0)),
    )
    cv2.imwrite(
        "graphcam_vis/{}_{}_ori.png".format(file_name, site),
        np.rot90(output_img, axes=(1, 0)),
    )

    # for i in range(1, len(vis_all)):
    cv2.imwrite(
        f"graphcam_vis/{file_name}_{site}_cam_{p_pred}.png", np.rot90(vis, axes=(1, 0))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphCAM")
    parser.add_argument(
        "--path_file",
        type=str,
        default="test.txt",
        help="txt file contains test sample",
    )
    parser.add_argument("--path_patches", type=str, default="", help="")
    parser.add_argument("--path_WSI", type=str, default="", help="")
    parser.add_argument("--path_graph", type=str, default="", help="")

    args = parser.parse_args()
    main(args)
