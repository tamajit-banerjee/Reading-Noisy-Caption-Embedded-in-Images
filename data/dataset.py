import torch
import os
import cv2
import pandas as pd

def generate_vocabulary(ann_file, vocab_path=None):
    vocab = ["<start>", "<end>", "<unk>"]
    if vocab_path is not None and os.path.exists(vocab_path):
        df = pd.read_csv(vocab_path)
        return df.iloc[:, 0].values
    df = pd.read_csv(ann_file, header=None, delimiter="\t")
    for caption in df.iloc[:, 1].values:
        vocab += caption.split(" ")
    vocab = list(set(vocab))
    vocab = [w.lower() for w in vocab]
    if vocab_path is not None:
        pd.DataFrame({"vocab": vocab}).to_csv(vocab_path, index=False)
    return vocab


class CaptionDataset():
    def __init__(self, img_prefix, ann_file=None, img_transforms=None, cap_transforms=None, vocab_path=None):
        super().__init__()
        self.img_prefix = img_prefix
        self.ann_file = ann_file
        self.data_idx = None
        self.img_transforms = img_transforms
        self.cap_transforms = cap_transforms
        self.vocab_path = vocab_path
        self.parse_annotations()
    
    def parse_annotations(self):
        if self.ann_file is not None:
            self.data_idx = []
            df = pd.read_csv(self.ann_file, header=None, delimiter="\t")
            self.vocab = generate_vocabulary(self.ann_file, self.vocab_path)
            self.data_idx = [ 
                {
                    "image_name": df.iloc[:, 0].values[i],
                    "caption": df.iloc[:, 1].values[i],
                }
                for i in range(len(df.iloc[:, 0].values))
            ]   
        else:
            self.data_idx = []
            for img_name in os.listdir(self.img_prefix):
                if "DS_Store" in img_name:
                    continue
                self.data_idx.append({
                    "image_name": img_name,
                    "caption": None,
                })
        
    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, index):
        data = self.data_idx[index]
        img_name = data["image_name"]
        caption = data["caption"]
        
        img_path = os.path.join(self.img_prefix, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.FloatTensor(image)/255
        image = image.permute(2, 0, 1)

        if self.img_transforms is not None and callable(self.img_transforms):
            image = self.img_transforms(image)
        
        if self.cap_transforms is not None and callable(self.cap_transforms) and caption is not None:
            caption = self.cap_transforms(caption)

        if self.ann_file is not None:
            img_name_write = img_name
        else:
            img_name_write = self.img_prefix.split(os.path.sep)
            img_name_write = [i for i in img_name_write if i!=""]
            img_name_write = os.path.join(img_name_write[-1], img_name)
        return image, caption, img_name_write


