"""
Include DECREE's trigger inversion and score assigning stages
"""

import os

os.environ["HF_HOME"] = os.path.abspath(
    "/data/gpfs/projects/punim1623/DECREE/external_clip_models"
)
import argparse, random, time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import open_clip
from models import get_encoder_architecture_usage
from utils.utils import assert_range, epsilon, compute_self_cos_sim
from imagenet import get_processing, getTensorImageNet, _mean, _std
from datetime import datetime
from utils.encoders import (
    pretrained_clip_sources,
    process_decree_encoder,
    process_hanxun_encoder,
    process_openclip_encoder,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
timestamp = (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(random.randint(0, 100))
    + "_"
    + str(random.randint(0, 100))
)

"""
Helper function for computing LID
"""


def lid_mle(data, reference, k=20, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    b = data.shape[0]
    k = min(k, b - 2)

    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)

    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    a, idx = torch.sort(r, dim=1)
    lids = -k / torch.sum(torch.log(a[:, 1:k] / a[:, k].view(-1, 1) + 1.0e-4), dim=1)
    return lids  # [bs,]


"""
print out detection performance scores
"""


def finalize(
    args,
    fp,
    id,
    train_mask_tanh,
    train_patch_tanh,
    clean_train_loader,
    model,
    test_transform,
    gt,
    regular_best=1,
    clean_unnormalized_L1_norm_max=1,
):

    train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
    train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)

    l2_dist_quantile_normalized = calculate_distance_metric(
        args.eval_metric,
        clean_train_loader,
        train_mask_tanh,
        train_patch_tanh,
        model,
        DEVICE,
        test_transform,
        args.quantile_low,
        args.quantile_high,
    )

    if not args.learned_trigger_folder:
        # save trigger locally
        inv_trigger_save_file_name = f"trigger_inv_{timestamp}/{id}"
        torch.save(train_mask_tanh, inv_trigger_save_file_name + "_inv_trigger_mask.pt")
        torch.save(
            train_patch_tanh, inv_trigger_save_file_name + "_inv_trigger_patch.pt"
        )

    result = f"{id},{gt},{regular_best/clean_unnormalized_L1_norm_max:.4f},{l2_dist_quantile_normalized:.4f}\n"

    print(result)

    fp.write(result)
    fp.flush()  # Manually flush after each write


"""
Obtain clean quantiles (used before training, if l2_norm loss is used in training)
"""


# def get_clean_quantile_range(
#     clean_train_loader, model, test_transform, quantile_low, quantile_high
# ):
#     model.eval()
#     clean_out_all = []
#     for clean_x_batch, _ in clean_train_loader:
#         clean_x_batch = clean_x_batch.to(DEVICE)
#         clean_input = torch.stack(
#             [test_transform(img.permute(2, 0, 1) / 255.0) for img in clean_x_batch]
#         )
#         clean_input = clean_input.to(dtype=torch.float).to(DEVICE)
#         with torch.no_grad():
#             clean_out = model(clean_input)  # [bs, 1024]
#             clean_out_all.append(clean_out)

#     clean_out_all = torch.cat(clean_out_all, dim=0)  # [total, 1024]
#     clean_quantile_start = torch.quantile(
#         clean_out_all, q=quantile_low, dim=0
#     )  # [1024, ]
#     clean_quantile_end = torch.quantile(
#         clean_out_all, q=quantile_high, dim=0
#     )  # [1024, ]
#     clean_quantile_range = clean_quantile_end - clean_quantile_start + epsilon()

#     return clean_quantile_range


"""
calculate distance
Output: single value
"""


def calculate_distance_metric(
    our_metric,
    clean_train_loader,
    mask,
    patch,
    model,
    DEVICE,
    test_transform,
    quantile_low,
    quantile_high,
):
    # fusion = mask * patch.detach()  # (0, 255), [h, w, 3]
    model.eval()

    clean_out_all, bd_out_all = [], []

    # each batch
    for clean_x_batch, _ in clean_train_loader:
        clean_x_batch = clean_x_batch.to(DEVICE)

        bd_x_batch = (1 - mask) * clean_x_batch + mask * patch
        bd_x_batch = torch.clip(bd_x_batch, min=0, max=255)

        clean_input = torch.stack(
            [test_transform(img.permute(2, 0, 1) / 255.0) for img in clean_x_batch]
        )
        bd_input = torch.stack(
            [test_transform(img.permute(2, 0, 1) / 255.0) for img in bd_x_batch]
        )
        clean_input = clean_input.to(dtype=torch.float).to(DEVICE)
        bd_input = bd_input.to(dtype=torch.float).to(DEVICE)

        # extract the visual representations
        with torch.no_grad():
            clean_out = model(
                clean_input
            )  # [bs, 1024], value range may depend on visual encoder's arch
            bd_out = model(bd_input)

            clean_out_all.append(clean_out)
            bd_out_all.append(bd_out)

    clean_out_all = torch.cat(clean_out_all, dim=0)  # [total, 1024]
    bd_out_all = torch.cat(bd_out_all, dim=0)  # [total, 1024]

    if our_metric == "l2":
        l2_dist = torch.norm((clean_out_all - bd_out_all), dim=1).detach().tolist()
        l2_dist = np.mean(l2_dist)
        return l2_dist
    elif our_metric == "l2_norm":
        clean_quantile_start = torch.quantile(
            clean_out_all, q=quantile_low, dim=0
        )  # [1024, ]
        clean_quantile_end = torch.quantile(
            clean_out_all, q=quantile_high, dim=0
        )  # [1024, ]
        clean_quantile_range = clean_quantile_end - clean_quantile_start + epsilon()

        l2_dist_quantile_normalized = (
            torch.norm((clean_out_all - bd_out_all) / clean_quantile_range, dim=1)
            .detach()
            .tolist()
        )
        l2_dist_quantile_normalized = np.mean(l2_dist_quantile_normalized)

        return l2_dist_quantile_normalized
    elif our_metric == "lid":
        lids = lid_mle(bd_out_all, bd_out_all).detach().tolist()
        return np.mean(lids)
    elif our_metric == "lid_on_clean":
        lids = lid_mle(clean_out_all, clean_out_all).detach().tolist()
        return np.mean(lids)
    elif our_metric == "cos_sim":
        cos_sim = (
            F.cosine_similarity(clean_out_all.flatten(1), bd_out_all.flatten(1), dim=1)
            .detach()
            .tolist()
        )  # [bs]
        return np.mean(cos_sim)

    # TODO: other metrics


"""
adjust learning rate
"""


def adjust_learning_rate(optimizer, epoch, args):

    thres = [200, 500]

    if epoch < thres[0]:  # 200
        lr = args.lr  # 0.5
    elif epoch < thres[1]:  # 500
        lr = 0.1
    else:
        lr = 0.05
    # print("epoch: {}  lr: {:.4f}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main(args, model_source, gt, id, encoder_path, fp):

    print(f">>> now processing {model_source} {id}")

    """
    Load Model
    """

    if model_source == "decree":
        model_ckpt_path = encoder_path
        model_ckpt = torch.load(model_ckpt_path, map_location=DEVICE)
        load_model = get_encoder_architecture_usage(args).to(DEVICE)
        load_model.visual.load_state_dict(model_ckpt["state_dict"])
        mask_size = 224
    elif model_source == "hanxun":
        model_ckpt_path = encoder_path
        load_model, _, _ = open_clip.create_model_and_transforms(encoder_path)
        load_model = load_model.to(DEVICE)
        mask_size = 224
    elif model_source == "openclip":
        (model_name, pretrained_key) = encoder_path
        model_ckpt_path = model_name + "_" + pretrained_key
        load_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_key
        )
        load_model = load_model.to(DEVICE)
        mask_size = load_model.visual.image_size
        # TODO: remove later
        print(mask_size, type(mask_size))
    elif model_source == "openclip_backdoored":
        (bd_model_path, arch, key) = encoder_path
        load_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=key)

        mask_size = load_model.visual.image_size
        # TODO: remove later
        print(mask_size, type(mask_size))
        # load ckpt
        bd_model_ckpt = torch.load(bd_model_path, map_location=DEVICE)
        load_model.visual.load_state_dict(bd_model_ckpt)
        load_model = load_model.to(DEVICE)

    model = load_model.visual
    model.eval()  # the poisoned/clean CLIP; we only need it to generate representations

    """
    Prepare Trigger
    """
    if args.learned_trigger_folder:
        learned_mask = torch.load(
            os.path.join(args.learned_trigger_folder, f"{id}_inv_trigger_mask.pt"),
            map_location=DEVICE,
        )
        learned_patch = torch.load(
            os.path.join(args.learned_trigger_folder, f"{id}_inv_trigger_patch.pt"),
            map_location=DEVICE,
        )
    else:
        ### This old code is not efficient, as we don't really the trigger_file, but only the trigger size
        # trigger_file = "trigger/trigger_pt_white_185_24.npz"
        # trigger_mask, trigger_patch = None, None
        # with np.load(trigger_file) as data:
        #     trigger_mask = np.reshape(data["tm"], (mask_size, mask_size, 3))
        #     trigger_patch = np.reshape(
        #         data["t"], (mask_size, mask_size, 3)
        #     )  # .astype(np.uint8)

        # train_mask_2d = torch.rand(trigger_mask.shape[:2], dtype=torch.float64).to(
        #     DEVICE
        # )  # [h, w]
        # train_patch = torch.rand_like(
        #     torch.tensor(trigger_patch), dtype=torch.float64
        # ).to(DEVICE)

        train_mask_2d = torch.rand((mask_size, mask_size), dtype=torch.float64).to(
            DEVICE
        )  # [h, w]
        train_patch = torch.rand((mask_size, mask_size, 3), dtype=torch.float64).to(
            DEVICE
        )

        # Purpose: optimization in a bounded range is hard for gradient-based methods. By using arctanh, you convert the variable into an unconstrained space (so the optimizer can freely adjust any real value). During forward passes, you can map it back to [0,1] using the hyperbolic tangent: x = (torch.tanh(z) + 1) / 2
        train_mask_2d = torch.arctanh((train_mask_2d - 0.5) * (2 - epsilon()))
        train_patch = torch.arctanh((train_patch - 0.5) * (2 - epsilon()))
        train_mask_2d.requires_grad = True
        train_patch.requires_grad = True

    """
    Prepare Dataloader
    """
    test_transform = transforms.Compose(
        [transforms.Normalize(_mean["imagenet"], _std["imagenet"])]
    )
    pre_transform, _ = get_processing(
        "imagenet", augment=False, is_tensor=False, need_norm=False, size=mask_size
    )  # get un-normalized tensor
    clean_train_data = getTensorImageNet(
        pre_transform
    )  # when later get_item, the returned image is in range [0, 255] and shape (H,W,C)
    clean_train_data.rand_sample(0.2)
    clean_train_loader = DataLoader(
        clean_train_data, batch_size=args.batch_size, pin_memory=True, shuffle=True
    )

    if args.learned_trigger_folder:
        #  use previously saved triggers
        finalize(
            args,
            fp,
            id,
            learned_mask,
            learned_patch,
            clean_train_loader,
            model,
            test_transform,
            gt,
        )
    else:
        """
        Prepare Optimizer
        """
        optimizer = torch.optim.Adam(
            params=[train_mask_2d, train_patch], lr=args.lr, betas=(0.5, 0.9)
        )

        """
        Loss and weights
        """
        loss_cos, loss_reg = None, None
        init_loss_lambda = 1e-3
        loss_lambda = init_loss_lambda  # balance between loss_cos and loss_reg
        adaptor_lambda = 5.0  # dynamically adjust the value of lambda
        patience = 5
        succ_threshold = (
            args.thres
        )  # cos-loss threshold for a successful reversed trigger, 0.99
        epochs = 1000

        # early stop
        regular_best = 1 / epsilon()
        early_stop_reg_best = regular_best
        early_stop_cnt = 0
        early_stop_patience = None  # 2 * patience

        # adjust for lambda
        # adaptor_up_flag, adaptor_down_flag = False, False
        adaptor_up_cnt, adaptor_down_cnt = 0, 0
        lambda_set_cnt = 0
        lambda_set_patience = 2 * patience
        lambda_min = 1e-7
        early_stop_patience = 7 * patience  # 35

        # print(
        #     f"Config: lambda_min: {lambda_min}, "
        #     f"adapt_lambda: {adaptor_lambda}, "
        #     f"lambda_set_patience: {lambda_set_patience},"
        #     f"succ_threshold: {succ_threshold}, "
        #     f"early_stop_patience: {early_stop_patience},"
        # )

        regular_list, cosine_list = [], []

        clean_unnormalized = []  # storing clean image's information

        # calculate the quantile values beforehand
        # clean_quantile_range = get_clean_quantile_range(
        #     clean_train_loader,
        #     model,
        #     test_transform,
        #     args.quantile_low,
        #     args.quantile_high,
        # )

        for e in range(epochs):

            # adjust learning rate based on current epoch
            adjust_learning_rate(optimizer, e, args)

            res_best = {"mask": None, "patch": None}
            loss_list = {"loss": [], "cos": [], "reg": []}

            # each batch
            for step, (clean_x_batch, _) in enumerate(clean_train_loader):

                # assert image is valid
                assert "Tensor" in clean_x_batch.type()  # no transform inside loader
                assert clean_x_batch.shape[-1] == 3
                clean_x_batch = clean_x_batch.to(DEVICE)
                assert_range(clean_x_batch, 0, 255)

                # reverse range and clip values
                train_mask_3d = train_mask_2d.unsqueeze(2).repeat(
                    1, 1, 3
                )  # shape(H,W)->(H,W,1)->(H,W,3)
                train_mask_tanh = (
                    torch.tanh(train_mask_3d) / (2 - epsilon()) + 0.5
                )  # range-> (0, 1)
                train_patch_tanh = (
                    torch.tanh(train_patch) / (2 - epsilon()) + 0.5
                ) * 255  # -> (0, 255)
                train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
                train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)

                # create poisoned image
                bd_x_batch = (
                    1 - train_mask_tanh
                ) * clean_x_batch + train_mask_tanh * train_patch_tanh
                bd_x_batch = torch.clip(
                    bd_x_batch, min=0, max=255
                )  # .to(dtype=torch.uint8)

                # test_transform
                bd_input = []
                # clean_input = []

                # each clean input in the current batch
                for i in range(clean_x_batch.shape[0]):
                    if e == 0:
                        clean_unnormalized.append(clean_x_batch[i])
                    bd_input.append(
                        test_transform(bd_x_batch[i].permute(2, 0, 1) / 255.0)
                    )
                    # clean_input.append(
                    #     test_transform(clean_x_batch[i].permute(2, 0, 1) / 255.0)
                    # )

                bd_input = torch.stack(bd_input)
                assert_range(bd_input, -3, 3)
                bd_input = bd_input.to(dtype=torch.float).to(DEVICE)

                # clean_input = torch.stack(clean_input)
                # clean_input = clean_input.to(dtype=torch.float).to(DEVICE)

                # extract the visual representations
                bd_out = model(bd_input)  # [bs, 1024]
                # clean_out = model(clean_input)  # [bs, 1024]

                # loss calculation
                loss_cos = -compute_self_cos_sim(
                    bd_out
                )  # average of pairwise similarity
                loss_reg = torch.sum(torch.abs(train_mask_tanh))  # L1 norm
                # TODO[DONE]: uncomment this and above to add loss_l2_dist
                # loss_l2_dist = torch.norm(
                #     (clean_out - bd_out) / clean_quantile_range, dim=1
                # ).mean()
                loss = (
                    loss_cos
                    + loss_reg * loss_lambda
                    # - loss_l2_dist * args.coeff_l2_dist
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list["loss"].append(loss)
                loss_list["cos"].append(loss_cos)
                loss_list["reg"].append(loss_reg)

                # record the best result so far
                if (torch.abs(loss_cos) > succ_threshold) and (loss_reg < regular_best):
                    train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
                    train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)
                    res_best["mask"] = train_mask_tanh
                    res_best["patch"] = train_patch_tanh
                    regular_best = loss_reg

                # check for early stop. The early_stop_cnt records the count of consecutive times that early-stop criterion is met
                if regular_best < 1 / epsilon():  # an valid trigger has been found
                    if regular_best >= early_stop_reg_best:
                        early_stop_cnt += 1
                    else:
                        early_stop_cnt = 0
                early_stop_reg_best = min(regular_best, early_stop_reg_best)

                # adjust loss_lambda
                if loss_lambda < lambda_min and (torch.abs(loss_cos) > succ_threshold):
                    lambda_set_cnt += 1
                    if lambda_set_cnt > lambda_set_patience:
                        loss_lambda = init_loss_lambda
                        adaptor_up_cnt, adaptor_down_cnt = 0, 0
                        # adaptor_up_flag, adaptor_down_flag = False, False
                        # print("Initialize lambda to {loss_lambda}")
                else:
                    lambda_set_cnt = 0

                if torch.abs(loss_cos) > succ_threshold:
                    adaptor_up_cnt += 1
                    adaptor_down_cnt = 0
                else:
                    adaptor_down_cnt += 1
                    adaptor_up_cnt = 0

                if adaptor_up_cnt > patience:
                    if loss_lambda < 1e5:
                        loss_lambda *= adaptor_lambda
                    adaptor_up_cnt = 0
                    # adaptor_up_flag = True
                    # print(f"step{step}:loss_lambda is up to {loss_lambda}")
                elif adaptor_down_cnt > patience:
                    if loss_lambda >= lambda_min:
                        loss_lambda /= adaptor_lambda
                    adaptor_down_cnt = 0
                    # adaptor_down_flag = True
                    # print(f"step{step}:loss_lambda is down to {loss_lambda}")

            # calculate max L1-norm of clean images
            if e == 0:
                clean_unnormalized = torch.stack(clean_unnormalized)  # [bs, h, w, 3]
                clean_unnormalized = clean_unnormalized.to(dtype=torch.float).to(DEVICE)
                # [bs, ]
                clean_unnormalized_L1_norm = torch.sum(
                    torch.abs(clean_unnormalized), dim=0
                )
                # max L1-norm
                clean_unnormalized_L1_norm_max = torch.max(clean_unnormalized_L1_norm)

            # loss_avg_e = torch.mean(torch.stack((loss_list["loss"])))
            loss_cos_e = torch.mean(torch.stack((loss_list["cos"])))
            # loss_reg_e = torch.mean(torch.stack((loss_list["reg"])))

            # print average loss values for the current epoch
            # print(
            #     f"e={e}, loss={loss_avg_e:.6f}, loss_cos={loss_cos_e:.6f}, "
            #     f"loss_reg={loss_reg_e:.6f}, cur_reg_best={regular_best:.6f}, "
            #     f"es_reg_best:{early_stop_reg_best:.6f}"
            # )

            # record the average losses over all the epochs so far
            # regular_list.append(str(round(float(loss_reg_e), 2)))
            # cosine_list.append(str(round(float(-loss_cos_e), 2)))

            # save images
            # if res_best["mask"] != None and res_best["patch"] != None:
            #     assert_range(res_best["mask"], 0, 1)
            #     assert_range(res_best["patch"], 0, 255)

            #     fusion = np.asarray(
            #         (res_best["mask"] * res_best["patch"]).detach().cpu(), np.uint8
            #     )
            #     mask = np.asarray(res_best["mask"].detach().cpu() * 255, np.uint8)
            #     patch = np.asarray(res_best["patch"].detach().cpu(), np.uint8)
            #     # fusion = (mask / 255.0 * patch).astype(np.uint8)

            #     dir = f"trigger_inv_{timestamp}/{id}"
            #     if not os.path.exists(f"{dir}"):
            #         os.makedirs(f"{dir}")

            #     suffix = f"e{e}_reg{regular_best:.2f}"
            #     mask_img = Image.fromarray(mask).save(f"{dir}/mask_{suffix}.png")
            #     patch_img = Image.fromarray(patch).save(f"{dir}/patch_{suffix}.png")
            #     fusion_img = Image.fromarray(fusion).save(f"{dir}/fus_{suffix}.png")

            # meet the final early-stop criterion: (1) average cos_sim is larger than succ_threshold; (2) early_stop_cnt surpasses the patience
            if (
                torch.abs(loss_cos_e) > succ_threshold
                and early_stop_cnt > early_stop_patience
            ):
                print("Early stop!")

                # print(f"End: {duration:.4f}s")
                # print(f"model_ckpt_path: {model_ckpt_path}")
                # print(f"L1: {regular_best:.4f}")
                # print(
                #     "reg:", ",".join(regular_list)
                # )  # the average L1 reg loss over all the epochs so far
                # print(
                #     "cos:", ",".join(cosine_list)
                # )  # the average cos_sim loss over all the epochs so far
                # print(f"clean_unnormalized_L1_norm_max: {clean_unnormalized_L1_norm_max}")
                # print(f"clean_normalized_L1_norm_max: {clean_normalized_L1_norm_max}")

                finalize(
                    args,
                    fp,
                    id,
                    train_mask_tanh,
                    train_patch_tanh,
                    clean_train_loader,
                    model,
                    test_transform,
                    gt,
                    regular_best,
                    clean_unnormalized_L1_norm_max,
                )
                return

        finalize(
            args,
            fp,
            id,
            train_mask_tanh,
            train_patch_tanh,
            clean_train_loader,
            model,
            test_transform,
            gt,
            regular_best,
            clean_unnormalized_L1_norm_max,
        )
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect bd in encoder")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Number of images in each mini-batch",
    )
    parser.add_argument(
        "--lr", default=0.5, type=float, help="learning rate on trigger"
    )
    parser.add_argument(
        "--coeff_l2_dist", default=0, type=float, help="coefficient for l2_dist loss"
    )
    parser.add_argument("--seed", default=80, type=int, help="random seed")
    parser.add_argument(
        "--result_file",
        default=f"results_{timestamp}.txt",
        type=str,
        help="result file",
    )
    parser.add_argument("--thres", default=0.99, type=float, help="success threshold")
    parser.add_argument(
        "--external_clip_store_folder",
        default="./external_clip_models",
        type=str,
        help="where to store clips models sourced from public",
    )
    parser.add_argument(
        "--learned_trigger_folder",
        type=str,
        help="saved trigger fodler",
    )
    parser.add_argument(
        "--quantile_low",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "--quantile_high",
        default=0.95,
        type=float,
    )
    parser.add_argument(
        "--note",
        type=str,
        help="note to help identify experiment",
    )
    parser.add_argument(
        "--eval_metric",
        type=str,
        choices=[
            "l2",
            "l2_norm",
            "lid",
            "lid_on_clean",
            "cos_sim",
        ],  # TODO: other metrics
        default="l2_norm",
        help="our evaluation metric",
    )
    args = parser.parse_args()
    print(args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trigger_save_dir = f"trigger_inv_{timestamp}"
    if not os.path.exists(trigger_save_dir):
        os.makedirs(trigger_save_dir)

    if not os.path.exists(args.external_clip_store_folder):
        os.makedirs(args.external_clip_store_folder)

    fp = open(args.result_file, "a")

    # for encoder in pretrained_clip_sources["decree"]:
    #     encoder_info = process_decree_encoder(encoder)

    #     main(
    #         args,
    #         "decree",
    #         encoder_info["gt"],
    #         encoder_info["id"],
    #         encoder_info["path"],
    #         fp,
    #     )

    # for encoder in pretrained_clip_sources["hanxun"]:
    #     encoder_info = process_hanxun_encoder(encoder)

    #     main(
    #         args,
    #         "hanxun",
    #         encoder_info["gt"],
    #         encoder_info["id"],
    #         encoder_info["path"],
    #         fp,
    #     )

    for encoder in pretrained_clip_sources["openclip"]:
        encoder_info = process_openclip_encoder(encoder)
        arch = encoder_info["arch"]

        # # TODO[DONE]: use different coeff_l2_dist for VIT or Resnet
        # if "vit" in arch.lower():
        #     args.coeff_l2_dist = 0.001
        # elif "rn" in arch.lower():
        #     args.coeff_l2_dist = 0.0001
        # else:
        #     raise Exception("Unknown model architecture")

        # TODO: remove this later
        if arch != "RN50x4":
            continue

        main(
            args,
            "openclip",
            encoder_info["gt"],
            encoder_info["id"],
            (arch, encoder_info["key"]),
            fp,
        )

    saved_encoders_folder = "saved_openclip_bd_encoders_all"
    for trigger in os.listdir(saved_encoders_folder):
        trigger_folder = os.path.join(saved_encoders_folder, trigger)

        if os.path.isdir(trigger_folder):
            for encoder_name in os.listdir(trigger_folder):

                # TODO: remove this later
                if "RN50x4" not in encoder_name and "ftrojan" not in encoder_name:
                    continue

                encodeer_filepath = os.path.join(
                    trigger_folder, encoder_name
                )  # the full path for each encodeer

                name_split = encoder_name.split("_")
                arch = name_split[1]
                key = "_".join(name_split[2:-6])
                trainset_percent = name_split[-3]
                ep = name_split[-1].split(".")[0]
                id = f"OPENCLIP_BD_{trigger}_trainsetp_{trainset_percent}_epoch_{ep}_{arch}_{key}"

                encoder_path = os.path.join(trigger_folder, encoder_name)

                # # TODO[DONE]: use different coeff_l2_dist for VIT or Resnet
                # if "vit" in arch.lower():
                #     args.coeff_l2_dist = 0.001
                # elif "rn" in arch.lower():
                #     args.coeff_l2_dist = 0.0001
                # else:
                #     raise Exception("Unknown model architecture")

                main(args, "openclip_backdoored", 1, id, (encoder_path, arch, key), fp)

    fp.close()
