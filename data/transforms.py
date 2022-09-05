import cv2
import pandas as pd
import torch
from PIL import Image
import cv2
import numpy as np

def get_caption_transforms(vocab_file_path):

    df = pd.read_csv(vocab_file_path)
    vocab = df.iloc[:, 0].values
    word_to_idx = {vocab[i]: i for i in range(len(vocab))}

    def tokenize(caption):
        word_list =  caption.split(" ")
        return [w.lower() for w in word_list]

    def add_start_end(word_list):
        return ["<start>"] + word_list + ["<end>"]

    def get_indices(word_list):
        return torch.tensor([word_to_idx[w] if w in vocab else word_to_idx["<unk>"] for w in word_list])

    def composed_transform(caption):
        return get_indices(add_start_end(tokenize(caption)))

    return composed_transform

def get_img_transforms(output_size):
    output_size = [3]+list(output_size)

    def pad_resize(image):
        out_img = torch.zeros(output_size)

        img_size = image.size()

        if img_size[1]/output_size[1] > img_size[2]/output_size[2]:
            resized_size = (3, output_size[1], int(img_size[2] * output_size[1]/img_size[1]))
        else:
            resized_size = (3, int(output_size[1] * output_size[2]/img_size[2]), output_size[2])
        image = 255*image
        img_np = np.asarray(image.permute(1, 2, 0).cpu())
        img_np_resize = cv2.resize(img_np, (resized_size[2], resized_size[1]), interpolation=cv2.INTER_AREA)
        resized_img = torch.FloatTensor(img_np_resize)/255
        resized_img = resized_img.permute(2, 0, 1)
        
        start_id1 = output_size[1]//2 - resized_size[1]//2
        start_id2 = output_size[2]//2 - resized_size[2]//2
        out_img[:, start_id1:start_id1+resized_size[1], start_id2:start_id2+resized_size[2]] = resized_img

        return out_img

    return pad_resize


def get_sentence_decoder(vocab_file_path):
    df = pd.read_csv(vocab_file_path)
    vocab = df.iloc[:, 0].values
    idx_to_word = {i:vocab[i] for i in range(len(vocab))}

    def decoder(sentence_indices):
        words = [idx_to_word[i] for i in sentence_indices]
        sentence = ""
        for word in words:
            if word != "<start>" and word != "<end>" and word != "<unk>":
                sentence += word
                sentence += " "
        return sentence[:-1]

    return decoder