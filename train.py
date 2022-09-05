from models.capnet import CapNet
from models.capnet_comp import CapNetComp
from data.dataset import CaptionDataset, generate_vocabulary
from data.dataloader import CaptionsDataLoader
from data.transforms import get_img_transforms, get_caption_transforms

import numpy as np
import os
import torch
from tqdm import tqdm
import argparse
import logging

USE_GPU = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser(description='Caption Reader Training')
    parser.add_argument('--root', type=str, help='path to dataset images root directory', default="../", required=False)
    parser.add_argument('--ann', type=str, help='path to annotation file', default="../Train_text.tsv", required=False)
    parser.add_argument('--vocab_path', type=str, help='path to save vocabulary', default="../vocabulary.csv", required=False)
    parser.add_argument('--bs', type=int, help='batch size', default=32, required=False)
    parser.add_argument('--epoch', type=int, help='number of epochs to train', default=10, required=False)
    parser.add_argument('--ckpt_path', type=str, help='path to save checkpoints', default="./checkpoints", required=False)
    parser.add_argument('--seed', type=int, help='seed', default=0, required=False)
    parser.add_argument('--comp', dest='comp', action='store_true')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    vocab_path = args.vocab_path
    vocab = generate_vocabulary(args.ann, vocab_path)
    train_dataset = CaptionDataset(
        img_prefix=args.root,
        ann_file=args.ann,
        img_transforms=get_img_transforms(output_size=(256, 256)),
        cap_transforms=get_caption_transforms(vocab_file_path=vocab_path),
        vocab_path=vocab_path,
    )
    train_dataloader = CaptionsDataLoader(
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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    os.makedirs(args.ckpt_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.ckpt_path, 'train.log'), filemode='a', format='%(levelname)s | %(message)s', level=logging.INFO)
    logging.info("Start new training")
    no_batches = len(train_dataloader)
    for epoch in range(1, args.epoch+1):
        # batch_id = 0
        # for batch in tqdm(train_dataloader):
        for batch_id, batch in enumerate(train_dataloader):
            images, captions, _ = batch
            batch_images = torch.stack(images)

            lengths = [cap.size(0) for cap in captions]
            train = torch.zeros(len(captions), max(lengths)-1)
            for i, cap in enumerate(captions):
                train[i][:lengths[i]-1]=np.delete(cap.data, lengths[i]-1)
            train = train.to(dtype = torch.long)
            if USE_GPU:
                batch_images = batch_images.cuda()
                captions = [cap.cuda() for cap in captions]
                train = train.cuda()

            pred_captions = model(batch_images, train, lengths)


            packed_caps = torch.nn.utils.rnn.pack_padded_sequence(train, lengths, batch_first=True, enforce_sorted =False)
            top_k = packed_caps.data.size(0)
            loss = criterion(pred_captions[:top_k, :], packed_caps.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (batch_id+1)%100 == 0:
                to_print_str = "Batch {}/{} | Epoch {}/{} | Loss: {}".format(batch_id, no_batches, epoch, args.epoch, loss.item())
                print(to_print_str)
        to_print_str = "Epoch {}/{} | Loss: {}".format(epoch, args.epoch, loss.item())
        print(to_print_str)
        logging.info(to_print_str)
        torch.save(model.state_dict(), os.path.join(args.ckpt_path, "model_{}.pth".format(epoch)))

if __name__ == "__main__":
    main()