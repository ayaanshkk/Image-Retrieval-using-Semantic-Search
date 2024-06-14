from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import os

if __name__ == '__main__':
    fe = FeatureExtractor()

    #the directories
    vgg16_feature_dir = Path("./static/feature/vgg16")
    clip_feature_dir = Path("./static/feature/clip")
    vgg16_feature_dir.mkdir(parents=True, exist_ok=True)
    clip_feature_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(f"Processing {img_path}")

        # Extract and save VGG16 features
        vgg16_feature = fe.extract_image_vgg16(img=Image.open(img_path))
        vgg16_feature_path = vgg16_feature_dir / (img_path.stem + ".npy") 
        np.save(vgg16_feature_path, vgg16_feature)

        # Extract and save CLIP features
        clip_feature = fe.extract_image_clip(img=Image.open(img_path))
        clip_feature_path = clip_feature_dir / (img_path.stem + ".npy")
        np.save(clip_feature_path, clip_feature)
