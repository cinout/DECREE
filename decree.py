import argparse, os, random, time

os.environ["HF_HOME"] = os.path.abspath(
    "/data/gpfs/projects/punim1623/DECREE/external_clip_models"
)
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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def finalize(
    train_mask_tanh,
    train_patch_tanh,
    clean_train_loader,
    model,
    test_transform,
    encoder_path,
    gt,
    start_time,
    regular_best,
    clean_unnormalized_L1_norm_max,
    fp,
    id,
):
    end_time = time.time()
    duration = end_time - start_time

    train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
    train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)

    l2_dist_quantile_normalized = calculate_distance_metric(
        clean_train_loader,
        train_mask_tanh,
        train_patch_tanh,
        model,
        DEVICE,
        test_transform,
    )

    # save trigger locally
    inv_trigger_save_file_name = f"trigger_inv_{timestamp}/{id}"
    torch.save(train_mask_tanh, inv_trigger_save_file_name + "_inv_trigger_mask.pt")
    torch.save(train_patch_tanh, inv_trigger_save_file_name + "_inv_trigger_patch.pt")

    fp.write(
        f"{encoder_path},{gt},{duration:.4f},{regular_best/clean_unnormalized_L1_norm_max:.4f},{l2_dist_quantile_normalized:.4f}\n"
    )


def calculate_distance_metric(
    clean_train_loader, mask, patch, model, DEVICE, test_transform
):
    # fusion = mask * patch.detach()  # (0, 255), [h, w, 3]
    model.eval()

    # l2_dist = []  # (bs,)
    # cossim = []  # (bs,)

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

        """
        distance metrics option
        """

        # # use L2
        # l2_dist_batch = torch.norm(clean_out - bd_out, dim=1).detach().tolist()
        # l2_dist.extend(l2_dist_batch)

        # # use cosine similarity
        # cossim_batch = (
        #     F.cosine_similarity(clean_out.flatten(1), bd_out.flatten(1), dim=1)
        #     .detach()
        #     .tolist()
        # )
        # cossim.extend(cossim_batch)

    # l2_dist = np.mean(l2_dist)
    # cossim = np.mean(cossim)

    clean_out_all = torch.cat(clean_out_all, dim=0)  # [total, 1024]
    bd_out_all = torch.cat(bd_out_all, dim=0)

    clean_quantile_start = torch.quantile(clean_out_all, q=0.05, dim=0)  # [1024, ]
    clean_quantile_end = torch.quantile(clean_out_all, q=0.95, dim=0)  # [1024, ]
    clean_quantile_range = clean_quantile_end - clean_quantile_start + epsilon()
    l2_dist_quantile_normalized = (
        torch.norm((clean_out_all - bd_out_all) / clean_quantile_range, dim=1)
        .detach()
        .tolist()
    )
    l2_dist_quantile_normalized = np.mean(l2_dist_quantile_normalized)

    return l2_dist_quantile_normalized


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
    ### load model
    """

    if model_source == "decree":
        # load from local path
        model_ckpt_path = encoder_path
        model_ckpt = torch.load(model_ckpt_path, map_location=DEVICE)
        load_model = get_encoder_architecture_usage(args).to(DEVICE)
        load_model.visual.load_state_dict(model_ckpt["state_dict"])
    elif model_source == "hanxun":
        model_ckpt_path = encoder_path
        load_model, _, _ = open_clip.create_model_and_transforms(encoder_path)
        load_model = load_model.to(DEVICE)
    elif model_source == "openclip":
        model_name, pretrained_key = encoder_path.split("@")
        model_ckpt_path = model_name + "_" + pretrained_key
        load_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_key
        )
        load_model = load_model.to(DEVICE)
    trigger_file = "trigger/trigger_pt_white_185_24.npz"

    mask_size = 224

    ### initialize trigger
    trigger_mask, trigger_patch = None, None

    # used as trigger and mask shape reference
    with np.load(trigger_file) as data:
        trigger_mask = np.reshape(data["tm"], (mask_size, mask_size, 3))
        trigger_patch = np.reshape(
            data["t"], (mask_size, mask_size, 3)
        )  # .astype(np.uint8)

    train_mask_2d = torch.rand(trigger_mask.shape[:2], dtype=torch.float64).to(
        DEVICE
    )  # [h, w]
    train_patch = torch.rand_like(torch.tensor(trigger_patch), dtype=torch.float64).to(
        DEVICE
    )

    # what's the purpose of this?
    # (1) Shifts the data range from [0, 1] → [-0.5, 0.5]
    # (2) Scales it up by roughly 2, giving [-1, 1] but slightly within it: [-1 + ε, 1 - ε]. The epsilon() keeps the values strictly inside (-1, 1) to avoid undefined values
    # (3) Applies the inverse hyperbolic tangent, which maps (-1, 1)  →  (-∞, +∞)
    # Purpose: optimization in a bounded range is hard for gradient-based methods. By using arctanh, you convert the variable into an unconstrained space (so the optimizer can freely adjust any real value).
    # during forward passes, you can map it back to [0,1] using the hyperbolic tangent: x = (torch.tanh(z) + 1) / 2
    train_mask_2d = torch.arctanh((train_mask_2d - 0.5) * (2 - epsilon()))
    train_patch = torch.arctanh((train_patch - 0.5) * (2 - epsilon()))
    train_mask_2d.requires_grad = True
    train_patch.requires_grad = True

    ### prepare dataloader and model
    test_transform = transforms.Compose(
        [transforms.Normalize(_mean["imagenet"], _std["imagenet"])]
    )

    # resize, crop, and tensorize to (0,1) range
    pre_transform, _ = get_processing(
        "imagenet", augment=False, is_tensor=False, need_norm=False
    )  # get un-normalized tensor

    # when later get_item, the returned image is in range [0, 255] and shape (H,W,C)
    clean_train_data = getTensorImageNet(pre_transform)
    clean_train_data.rand_sample(0.2)

    model = load_model.visual

    clean_train_loader = DataLoader(
        clean_train_data, batch_size=args.batch_size, pin_memory=True, shuffle=True
    )

    # # projector is not used
    # projector = torch.rand([1, 512], dtype=torch.float64).to(DEVICE)
    # projector = F.normalize(projector, dim=-1)

    optimizer = torch.optim.Adam(
        params=[train_mask_2d, train_patch], lr=args.lr, betas=(0.5, 0.9)
    )

    # why set to eval()?, because model is the poisoned/clean CLIP, and we only need it to generate representations
    model.eval()

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
    adaptor_up_cnt, adaptor_down_cnt = 0, 0
    adaptor_up_flag, adaptor_down_flag = False, False
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
    start_time = time.time()

    # storing clean image's information
    clean_unnormalized, clean_normalized = [], []

    # each epoch
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

            # test_transform all clean and poisoned images
            bd_input = []
            # each clean input in the current batch
            for i in range(clean_x_batch.shape[0]):

                # value to (0,1) range, and then normalize by imagent mena and var
                clean_trans = test_transform(clean_x_batch[i].permute(2, 0, 1) / 255.0)

                if e == 0:
                    clean_unnormalized.append(clean_x_batch[i])
                    clean_normalized.append(clean_trans)

                bd_trans = test_transform(bd_x_batch[i].permute(2, 0, 1) / 255.0)
                bd_input.append(bd_trans)

            # clean_input_unnormalized = torch.stack(
            #     clean_input_unnormalized
            # )  # [bs, h, w, 3]
            # clean_input_normalized = torch.stack(clean_input_normalized)

            bd_input = torch.stack(bd_input)
            assert_range(bd_input, -3, 3)
            # assert_range(clean_input_normalized, -3, 3)

            # clean_input_normalized = clean_input_normalized.to(dtype=torch.float).to(
            #     DEVICE
            # )
            # clean_input_unnormalized = clean_input_unnormalized.to(
            #     dtype=torch.float
            # ).to(DEVICE)

            bd_input = bd_input.to(dtype=torch.float).to(DEVICE)

            # extract the visual representations
            bd_out = model(bd_input)

            ### extension for adaptive attack
            # projector = F.normalize(projector, dim=-1)
            # bd_out = projector * bd_out

            # loss calculation
            loss_cos = -compute_self_cos_sim(bd_out)  # average of pairwise similarity
            loss_reg = torch.sum(torch.abs(train_mask_tanh))  # L1 norm
            loss = loss_cos + loss_reg * loss_lambda

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
                    adaptor_up_flag, adaptor_down_flag = False, False
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
                adaptor_up_flag = True
                # print(f"step{step}:loss_lambda is up to {loss_lambda}")
            elif adaptor_down_cnt > patience:
                if loss_lambda >= lambda_min:
                    loss_lambda /= adaptor_lambda
                adaptor_down_cnt = 0
                adaptor_down_flag = True
                # print(f"step{step}:loss_lambda is down to {loss_lambda}")

        # calculate max L1 norm of clean inputs
        if e == 0:
            clean_unnormalized = torch.stack(clean_unnormalized)  # [bs, h, w, 3]
            clean_normalized = torch.stack(clean_normalized)

            clean_unnormalized = clean_unnormalized.to(dtype=torch.float).to(DEVICE)
            clean_normalized = clean_normalized.to(dtype=torch.float).to(DEVICE)

            # [bs, ]
            clean_unnormalized_L1_norm = torch.sum(torch.abs(clean_unnormalized), dim=0)
            clean_normalized_L1_norm = torch.sum(torch.abs(clean_normalized), dim=0)

            # max L1-norm
            clean_unnormalized_L1_norm_max = torch.max(clean_unnormalized_L1_norm)
            clean_normalized_L1_norm_max = torch.max(clean_normalized_L1_norm)

        loss_avg_e = torch.mean(torch.stack((loss_list["loss"])))
        loss_cos_e = torch.mean(torch.stack((loss_list["cos"])))
        loss_reg_e = torch.mean(torch.stack((loss_list["reg"])))

        # print average loss values for the current epoch
        # print(
        #     f"e={e}, loss={loss_avg_e:.6f}, loss_cos={loss_cos_e:.6f}, "
        #     f"loss_reg={loss_reg_e:.6f}, cur_reg_best={regular_best:.6f}, "
        #     f"es_reg_best:{early_stop_reg_best:.6f}"
        # )

        # record the average losses over all the epochs so far
        regular_list.append(str(round(float(loss_reg_e), 2)))
        cosine_list.append(str(round(float(-loss_cos_e), 2)))

        # save images
        if res_best["mask"] != None and res_best["patch"] != None:
            assert_range(res_best["mask"], 0, 1)
            assert_range(res_best["patch"], 0, 255)

            fusion = np.asarray(
                (res_best["mask"] * res_best["patch"]).detach().cpu(), np.uint8
            )
            mask = np.asarray(res_best["mask"].detach().cpu() * 255, np.uint8)
            patch = np.asarray(res_best["patch"].detach().cpu(), np.uint8)
            # fusion = (mask / 255.0 * patch).astype(np.uint8)

            dir = f"trigger_inv_{timestamp}/{id}"
            if not os.path.exists(f"{dir}"):
                os.makedirs(f"{dir}")

            suffix = f"e{e}_reg{regular_best:.2f}"
            mask_img = Image.fromarray(mask).save(f"{dir}/mask_{suffix}.png")
            patch_img = Image.fromarray(patch).save(f"{dir}/patch_{suffix}.png")
            fusion_img = Image.fromarray(fusion).save(f"{dir}/fus_{suffix}.png")

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
                train_mask_tanh,
                train_patch_tanh,
                clean_train_loader,
                model,
                test_transform,
                encoder_path,
                gt,
                start_time,
                regular_best,
                clean_unnormalized_L1_norm_max,
                fp,
                id,
            )

    finalize(
        train_mask_tanh,
        train_patch_tanh,
        clean_train_loader,
        model,
        test_transform,
        encoder_path,
        gt,
        start_time,
        regular_best,
        clean_unnormalized_L1_norm_max,
        fp,
        id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect bd in encoder")
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Number of images in each mini-batch",
    )
    parser.add_argument(
        "--lr", default=0.5, type=float, help="learning rate on trigger"
    )
    parser.add_argument("--seed", default=80, type=int, help="random seed")
    parser.add_argument("--mask_init", default="", type=str, help="init method of mask")
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
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
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

    # TODO: remove [:2]
    for encoder in pretrained_clip_sources["decree"][:2]:
        encoder_info = process_decree_encoder(encoder)

        main(
            args,
            "decree",
            encoder_info["gt"],
            encoder_info["id"],
            encoder_info["path"],
            fp,
        )

    for encoder in pretrained_clip_sources["hanxun"][:2]:
        encoder_info = process_hanxun_encoder(encoder)

        main(
            args,
            "hanxun",
            encoder_info["gt"],
            encoder_info["id"],
            encoder_info["path"],
            fp,
        )

    for encoder in pretrained_clip_sources["openclip"][:2]:
        encoder_info = process_openclip_encoder(encoder)

        main(
            args,
            "openclip",
            encoder_info["gt"],
            encoder_info["id"],
            encoder_info["arch"] + "@" + encoder_info["key"],
            fp,
        )

    fp.close()
