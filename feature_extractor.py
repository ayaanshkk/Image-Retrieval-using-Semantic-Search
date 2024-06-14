import numpy as np
from PIL import Image
import torch
import clip
from torchvision import models, transforms

class FeatureExtractor:
    def __init__(self):
        # Initialize VGG16 for image feature extraction
        self.vgg16 = models.vgg16(pretrained=True).features
        self.vgg16.eval()
        self.preprocess_vgg16 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Initialize CLIP for image and text feature extraction
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")

    def extract_image_vgg16(self, img):
        # Extract features using VGG16
        img_t = self.preprocess_vgg16(img).unsqueeze(0)
        with torch.no_grad():
            features = self.vgg16(img_t).flatten().numpy()
        features /= np.linalg.norm(features) 
        return features

    def extract_image_clip(self, img):
        # Extract features using CLIP
        img_t = self.clip_preprocess(img).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            features = self.clip_model.encode_image(img_t).cpu().numpy().flatten()
        features /= np.linalg.norm(features)  # Normalize features
        return features

    def extract_text(self, text):
        # Extract features using CLIP
        text_tokens = clip.tokenize([text]).to("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            features = self.clip_model.encode_text(text_tokens).cpu().numpy().flatten()
        features /= np.linalg.norm(features) 
        return features
