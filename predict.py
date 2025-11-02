import torch
import cv2
import argparse
import os
import numpy as np
import time
import copy
from tqdm import tqdm
from transformers import CLIPTokenizer
from torch.utils.data import DataLoader

import config
from model import ModulatedUNet
from dataset import CTDataset
from utils import dice_score, iou_score

def predict_single_image(model, image_path, prompt, device):
    dataset = CTDataset(
        data_dir=config.DATA_DIR,
        split='valid',
        text_model_name=config.TEXT_MODEL_NAME,
        image_size=config.IMAGE_SIZE
    )
    transform = dataset.transform
    tokenizer = CLIPTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
    text_tokens = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(image_tensor, text_tokens))
        mask = (preds > 0.5).cpu().numpy().squeeze().astype(np.uint8)

    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized * 255

def run_evaluation_loop(model, loader, device, desc, is_negative=False):
    model.eval()
    dice_list, iou_list, mean_pred, times = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if is_negative:
                masks = torch.zeros_like(masks)
            tokens = {k: v.to(device) for k, v in batch["text_tokens"].items()}

            start = time.time()
            preds = model(imgs, tokens)
            times.append((time.time() - start) / len(imgs))

            dice_list.append(dice_score(preds, masks))
            iou_list.append(iou_score(preds, masks))
            mean_pred.append((torch.sigmoid(preds) > 0.5).float().mean().item())

    return (
        np.mean(dice_list) if dice_list else 0.0,
        np.mean(iou_list) if iou_list else 0.0,
        np.mean(mean_pred),
        np.mean(times) if times else 0.0,
    )

def evaluate_on_test_set(model, device):
    print("\n--- Evaluating on Test Set ---")
    base_ds = CTDataset(config.DATA_DIR, 'test', config.TEXT_MODEL_NAME, config.IMAGE_SIZE)
    meta = base_ds.metadata

    def make_loader(prefix, prompt):
        ds = copy.deepcopy(base_ds)
        df = meta[meta['image_filename'].str.startswith(prefix)].drop_duplicates('image_filename').copy()
        df['prompt'] = prompt
        ds.metadata = df
        return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)

    loaders = {
        "crack_pos": make_loader('crack', "segment crack"),
        "taping_pos": make_loader('taping', "segment taping area"),
        "crack_neg": make_loader('crack', "segment taping area"),
        "taping_neg": make_loader('taping', "segment crack"),
    }

    c_dice, c_iou, _, c_time = run_evaluation_loop(model, loaders["crack_pos"], device, "Crack + Crack")
    t_dice, t_iou, _, t_time = run_evaluation_loop(model, loaders["taping_pos"], device, "Taping + Taping")
    _, _, n_c_mean, _ = run_evaluation_loop(model, loaders["crack_neg"], device, "Crack + Taping (Neg)", True)
    _, _, n_t_mean, _ = run_evaluation_loop(model, loaders["taping_neg"], device, "Taping + Crack (Neg)", True)

    print("\n--- Results ---")
    print(f"Crack (correct): Dice={c_dice:.4f}, IoU={c_iou:.4f}")
    print(f"Taping (correct): Dice={t_dice:.4f}, IoU={t_iou:.4f}")
    print(f"Crack (wrong prompt): MeanPred={n_c_mean:.6f}")
    print(f"Taping (wrong prompt): MeanPred={n_t_mean:.6f}")
    print(f"Avg Inference Time: {np.mean([c_time, t_time]) * 1000:.2f} ms")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["predict", "eval"])
parser.add_argument("--image", type=str)
parser.add_argument("--prompt", type=str)
parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pth.tar")
args = parser.parse_args()

model = ModulatedUNet(freeze_encoders=False).to(config.DEVICE)
ckpt = torch.load(args.checkpoint, map_location=config.DEVICE, weights_only=True)
model.load_state_dict(ckpt['state_dict'])
print(f"Loaded {args.checkpoint} (Epoch {ckpt.get('epoch', -1) + 1}) — {os.path.getsize(args.checkpoint)/(1024*1024):.2f} MB")

if args.mode == "predict":
    if not args.image or not args.prompt:
        raise ValueError("--image and --prompt are required in predict mode")
    mask = predict_single_image(model, args.image, args.prompt, config.DEVICE)
    os.makedirs("outputs/predictions", exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]
    slug = args.prompt.replace(" ", "_").replace("/", "_")
    out = f"outputs/predictions/{name}__{slug}.png"
    cv2.imwrite(out, mask)
    print(f"Saved prediction: {out}")

elif args.mode == "eval":
    evaluate_on_test_set(model, config.DEVICE)
