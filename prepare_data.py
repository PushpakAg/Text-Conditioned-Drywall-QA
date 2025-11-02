#%%
import os
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
#%%
raw_dir = "data/raw/"
processed_dir = "data/processed/"
logs_dir = "logs/"
train_r = 0.7
valid_r = 0.2
random_seed = 42
taping_prompts = [
    "segment taping area",
    "segment drywall tape",
    "segment wall joint",
    "segment drywall seam",
    "segment taped seam",
    "segment joint area",
    "segment tape line",
    "highlight drywall tape region",
    "find the taping compound area",
    "mask the drywall joint",
    "locate joint compound",
    "segment plastered seam",
    "segment taped joint between drywall sheets",
    "detect wall seam covered with tape",
    "identify drywall taping line",
    "segment area covered with drywall tape",
    "segment finishing tape area",
    "segment patched drywall joint",
    "segment gypsum board seam",
    "highlight taped connection"
]
crack_prompts = [
    "segment crack",
    "segment wall crack",
    "detect surface crack",
    "find cracks in wall",
    "highlight damaged area",
    "segment drywall crack",
    "segment surface defect",
    "segment fracture line",
    "locate structural crack",
    "mask the crack region",
    "segment hairline crack",
    "segment long wall crack",
    "segment vertical crack",
    "segment concrete crack",
    "detect fissure in wall",
    "identify crack on surface",
    "highlight fracture on drywall",
    "find wall defect line",
    "segment broken drywall area",
    "segment fine crack pattern"
]

os.makedirs(logs_dir, exist_ok=True)
log_path = os.path.join(logs_dir, "data_prepare_errors.txt")
#%%
def create_mask(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_node = root.find('size')
        width = int(size_node.find('width').text)
        height = int(size_node.find('height').text)
        mask = np.zeros((height, width), dtype=np.uint8)

        for obj in root.findall('object'):
            polygon_node = obj.find('polygon')
            if polygon_node is not None:
                points = []
                point_nodes = list(polygon_node)
                for i in range(0, len(point_nodes), 2):
                    x = float(point_nodes[i].text)
                    y = float(point_nodes[i+1].text)
                    points.append([x, y])

                if points:
                    points_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [points_np], color=(255))
                continue

            bndbox_node = obj.find('bndbox')
            if bndbox_node is not None:
                xmin = int(bndbox_node.find('xmin').text)
                ymin = int(bndbox_node.find('ymin').text)
                xmax = int(bndbox_node.find('xmax').text)
                ymax = int(bndbox_node.find('ymax').text)
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), color=(255), thickness=-1)
        return mask
    except Exception as e:
        with open(log_path, 'a') as f:
            f.write(f"Error processing {xml_path}: {e}\n")
        return None

def get_file_pairs(dataset_name, data_type):
    file_pairs = []
    dataset_path = os.path.join(raw_dir, dataset_name)
    for split in ["train", "valid", "test"]:
        annotation_dir = os.path.join(dataset_path, split)
        if not os.path.isdir(annotation_dir):
            continue
        for filename in os.listdir(annotation_dir):
            if filename.endswith('.xml'):
                xml_path = os.path.join(annotation_dir, filename)
                image_path = os.path.join(annotation_dir, filename.replace('.xml', '.jpg'))
                if os.path.exists(image_path):
                    file_pairs.append({
                        "image_path": image_path,
                        "xml_path": xml_path,
                        "type": data_type
                    })
    return file_pairs

def process(split_pairs, split_name):
    metadata_rows = []
    os.makedirs(os.path.join(processed_dir, split_name, "images"), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, split_name, "masks"), exist_ok=True)

    for pair in tqdm(split_pairs, desc=f"Processing {split_name} split"):
        data_type = pair['type']
        prompts = crack_prompts if data_type == 'crack' else taping_prompts
        mask_array = create_mask(pair['xml_path'])
        if mask_array is None:
            continue

        base_filename = os.path.basename(pair['image_path'])
        new_image_name = f"{data_type}_{base_filename}"
        new_mask_name = new_image_name.replace('.jpg', '.png')
        dest_image_path = os.path.join(processed_dir, split_name, "images", new_image_name)
        dest_mask_path = os.path.join(processed_dir, split_name, "masks", new_mask_name)

        shutil.copy(pair['image_path'], dest_image_path)
        cv2.imwrite(dest_mask_path, mask_array)

        for prompt in prompts:
            metadata_rows.append({
                "image_filename": new_image_name,
                "mask_filename": new_mask_name,
                "prompt": prompt
            })
    if metadata_rows:
        df = pd.DataFrame(metadata_rows)
        df.to_csv(os.path.join(processed_dir, split_name, 'metadata.csv'), index=False)
#%%
taping_pairs = get_file_pairs("Drywall-Join-Detect.v2i.voc", "taping")
crack_pairs = get_file_pairs("cracks.v1-tester.voc", "crack")
all_pairs = taping_pairs + crack_pairs
print(f"Total {len(all_pairs)} unique image/annotation pairs.")
random.seed(random_seed)
random.shuffle(all_pairs)
train_end_idx = int(len(all_pairs) * train_r)
valid_end_idx = train_end_idx + int(len(all_pairs) * valid_r)
train_pairs = all_pairs[:train_end_idx]
valid_pairs = all_pairs[train_end_idx:valid_end_idx]
test_pairs = all_pairs[valid_end_idx:]
process(train_pairs, "train")
process(valid_pairs, "valid")
process(test_pairs, "test")
print("Data preparation complete!")


#%%
def visualize_data_samples(processed_dir, split, num_samples=3):
    metadata_path = os.path.join(processed_dir, split, 'metadata.csv')
    image_dir = os.path.join(processed_dir, split, 'images')
    mask_dir = os.path.join(processed_dir, split, 'masks')
    df = pd.read_csv(metadata_path)
    unique_images = df.drop_duplicates(subset=['image_filename'])
    num_samples = min(num_samples, len(unique_images))
    random_samples = unique_images.sample(n=num_samples)
    for _, row in random_samples.iterrows():
        image_filename = row['image_filename']
        mask_filename = row['mask_filename']
        prompts_for_image = df[df['image_filename'] == image_filename]['prompt'].tolist()
        prompt_text = random.choice(prompts_for_image)
        image_path = os.path.join(image_dir, image_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"Warning: Could not load image or mask for {image_filename}")
            continue
        color_mask = np.zeros_like(image)
        color_mask[mask > 0] = [255, 0, 0]
        overlay = cv2.addWeighted(image, 1, color_mask, 0.4, 0)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Original Image', 'Ground Truth Mask', 'Overlay']
        images = [image, mask, overlay]
        for i, (ax, title, img) in enumerate(zip(axes, titles, images)):
            ax.imshow(img, cmap='gray' if i == 1 else None)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'Prompt: "{prompt_text}"', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

visualize_data_samples(
    processed_dir=processed_dir,
    split="train",
    num_samples=3
)
# %%
