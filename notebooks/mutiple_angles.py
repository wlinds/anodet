import os
import cv2
import torch
import matplotlib.pyplot as plt
from anodet import AnodetDataset, Padim, to_batch, classification, visualization
from torch.utils.data import DataLoader
from torchvision import transforms as T
import numpy as np

DATASET_PATH = os.path.realpath("../../_anodet/data/purple_duck")
MODEL_DATA_PATH = os.path.realpath("./distributions/p_duck")
THRESH = 13

def create_and_fit_padim(dataset_path, model_data_path, angle):
    dataset = AnodetDataset(os.path.join(dataset_path, f"train/good/cam_{angle}"))
    dataloader = DataLoader(dataset, batch_size=1)
    
    padim = Padim(backbone="resnet18")
    padim.fit(dataloader)
    
    torch.save(padim.mean, os.path.join(model_data_path, f"{angle}_mean.pt"))
    torch.save(padim.cov_inv, os.path.join(model_data_path, f"{angle}_cov_inv.pt"))

    return padim

def load_test_images(dataset_path, angle, image_paths):
    images = []
    for img_path in image_paths:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images

def process_test_images(padim, images, angle):
    batch = to_batch(images,
                     T.Compose([T.Resize(224),
                                T.CenterCrop(224),
                                T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                               ]),
                     torch.device('cpu')
                    )

    cov_inv_data = torch.load(os.path.join(MODEL_DATA_PATH, f"{angle}_cov_inv.pt"))
    mean_data = torch.load(os.path.join(MODEL_DATA_PATH, f"{angle}_mean.pt"))

    padim = Padim(backbone="resnet18", cov_inv=cov_inv_data, mean=mean_data, device=torch.device('cpu'))

    image_scores, score_maps = padim.predict(batch)
    score_map_classifications = classification(score_maps, THRESH)
    image_classifications = classification(image_scores, THRESH)
    
    return score_maps, score_map_classifications, image_scores, image_classifications

def visualize_and_save_images(set_idx, images, boundary_images, heatmap_images, highlighted_images):
    for idx in range(len(images)):
        fig, axs = plt.subplots(1, 4, figsize=(12, 6))
        fig.suptitle('Set {}: Image {}'.format(set_idx, idx), y=0.75, fontsize=14)
        axs[0].imshow(images[idx])
        axs[1].imshow(boundary_images[idx])
        axs[2].imshow(heatmap_images[idx])
        axs[3].imshow(highlighted_images[idx])
        plt.savefig(f'img_set{set_idx}_idx{idx}.png')
        #plt.show()


if __name__ == "__main__":
    angles = [2, 0, 1]
    for angle in angles:
        padim = create_and_fit_padim(DATASET_PATH, MODEL_DATA_PATH, angle)
        
        # Load test images for the current angle
        paths = {
            2: [
                os.path.join(DATASET_PATH, "test/albinism/cam_2_front/001.png"),
                os.path.join(DATASET_PATH, "test/albinism/cam_2_front/002.png")
            ],
            0: [
                os.path.join(DATASET_PATH, "test/albinism/cam_0_left/001.png"),
                os.path.join(DATASET_PATH, "test/albinism/cam_0_left/002.png")
            ],
            1: [
                os.path.join(DATASET_PATH, "test/albinism/cam_1_right/001.png"),
                os.path.join(DATASET_PATH, "test/albinism/cam_1_right/002.png")
            ]
        }
        
        test_images = load_test_images(DATASET_PATH, angle, paths[angle])
        score_maps, score_map_classifications, image_scores, image_classifications = process_test_images(padim, test_images, angle)
        
        visualize_and_save_images(angle, test_images, 
                                  visualization.framed_boundary_images(test_images, score_map_classifications, image_classifications, padding=40),
                                  visualization.heatmap_images(test_images, score_maps, alpha=0.5),
                                  visualization.highlighted_images(test_images, score_map_classifications, color=(128, 0, 128)))
    