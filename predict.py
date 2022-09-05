from models.capnet import CapNet
from models.capnet_comp import CapNetComp
from data.dataset import CaptionDataset, generate_vocabulary
from data.dataloader import CaptionsDataLoader
from data.transforms import get_img_transforms, get_caption_transforms, get_sentence_decoder
from beam_search import beam_search_pred

import pandas as pd
import os
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import torch
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

USE_GPU = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser(description='Caption Reader Prediction')
    parser.add_argument('--root', type=str, help='path to dataset images root directory', default="../", required=False)
    parser.add_argument('--ann', type=str, help='path to annotation file', default="../Train_text.tsv", required=False)
    parser.add_argument('--vocab_path', type=str, help='path to save vocabulary', default="../vocabulary.csv", required=False)
    parser.add_argument('--bs', type=int, help='batch size', default=1, required=False)
    parser.add_argument('--ckpt_path', type=str, help='path to load checkpoint from', default="./checkpoints/model_10.pth", required=False)
    parser.add_argument('--seed', type=int, help='seed', default=0, required=False)
    parser.add_argument('--save', type=str, help='path to save predictions', default="./output.tsv", required=False)
    parser.add_argument('--mode', type=str, help='test mode or dump mode', default="test", required=False)
    parser.add_argument('--comp', dest='comp', action='store_true')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    vocab_path = args.vocab_path
    vocab = generate_vocabulary(args.ann, vocab_path)
    word_to_idx = {vocab[i]: i for i in range(len(vocab))}
    if args.mode == "test":
        ann = args.ann 
    else:
        ann = None
    
    train_dataset = CaptionDataset(
        img_prefix=args.root,
        ann_file=ann,
        img_transforms=get_img_transforms(output_size=(256, 256)),
        cap_transforms=get_caption_transforms(vocab_file_path=vocab_path),
        vocab_path=vocab_path,
    )
    test_dataloader = CaptionsDataLoader(
        dataset=train_dataset,
        batch_size=args.bs,
        shuffle=True,
        seed=args.seed,
    )
    if args.comp:
        model = CapNetComp(
            embedding_dim=128,
            lstm_size=256,
            vocab_size=len(vocab),
            use_gpu=USE_GPU,
        )
    else:
        model = CapNet(
            embedding_dim=128,
            lstm_size=256,
            vocab_size=len(vocab),
            use_gpu=USE_GPU,
        )
    if USE_GPU:
        model = model.cuda()

    sentence_decoder = get_sentence_decoder(vocab_file_path=vocab_path)

    predictions_dict = {"name": [], "caption": []}
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    bleu_scores = []
    for batch_id, batch in enumerate(test_dataloader):
        images, cap, img_name = batch
        batch_images = torch.stack(images)

        if USE_GPU:
            batch_images = batch_images.cuda()
        
        pred_indices = beam_search_pred(batch_images, model, word_to_idx["<end>"], max_length=5)
        pred_caption = sentence_decoder(pred_indices)
        # print(pred_caption)
        
        if args.mode == "test":
            target = sentence_decoder(cap[0].numpy())
            bleu_score = sentence_bleu([target], pred_caption, weights=(1.0, 0, 0, 0))
            bleu_scores.append(bleu_score)

            print("{:5d}/{:5d} | BLEU score: {:.5f} | AVG: {:.5f}".format(batch_id, len(test_dataloader), bleu_score, sum(bleu_scores)/len(bleu_scores)),  end='\r')
        else:
            print("{:5d}/{:5d}".format(batch_id, len(test_dataloader)),  end='\r')
        predictions_dict["name"].append(img_name[0])
        predictions_dict["caption"].append(pred_caption)
    
        df = pd.DataFrame(predictions_dict)
        df.to_csv(args.save, sep="\t", index=False, header=False)
    if args.mode == "test":
        print("\nAverage BLEU score: ",  sum(bleu_scores)/(len(bleu_scores)))



if __name__ == "__main__":
    main()