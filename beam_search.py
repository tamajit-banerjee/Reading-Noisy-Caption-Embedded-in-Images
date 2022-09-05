import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

USE_GPU = torch.cuda.is_available()

def beam_search_pred(images, model, end_token, beam_size = 3, max_length=80):

    encoded_features = model.encoder(images).unsqueeze(1)
    if USE_GPU:
        encoded_features = encoded_features.cuda()
    hidden = None
    caption_word_id = []
    cur_pred = []
    next_seq = []
    
    output, hidden = model.decoder.forward_test(encoded_features)
    
    output = F.log_softmax(output, dim=1)
    prob, pred_ids = torch.topk(output, beam_size)
    prob = prob.cpu().detach().numpy().squeeze(0)
    pred_ids = pred_ids.cpu().detach().numpy().squeeze(0)
    for ids in range(len(pred_ids)):
        cur_pred.append([[pred_ids[ids]], prob[ids], hidden])

    while(len(caption_word_id) < max_length):
        for j in range(len(cur_pred)):
            cur_pred_torch = torch.tensor([cur_pred[j][0][-1]])
            cur_pred_list_torch = cur_pred[j][-1]
            if USE_GPU:
                cur_pred_torch = cur_pred_torch.cuda()
                cur_pred_list_torch = [x.cuda() for x in cur_pred_list_torch]
            output, hidden = model.decoder.get_bs_pred(cur_pred_torch, cur_pred_list_torch)
            output = F.log_softmax(output, dim=1)
            prob, pred_ids = torch.topk(output, beam_size)
            prob = prob.cpu().detach().numpy().squeeze(0)
            pred_ids = pred_ids.cpu().detach().numpy().squeeze(0)
            for ids in range(len(pred_ids)):
                next_seq.append([[cur_pred[j][0], [pred_ids[ids]]], (cur_pred[j][1] + prob[ids]), hidden])
                

        for seq_ in next_seq:
            seq_[0] = [item for sublist in seq_[0] for item in sublist]
            cur_pred.append(seq_)
        
        next_seq = []
        cur_pred = cur_pred[beam_size:]
        cur_pred = [c[1] for c in cur_pred]
        cur_pred = sorted(cur_pred, reverse=True)
        cur_pred = cur_pred[:beam_size]
        caption_word_id = cur_pred[0][0]

        if (caption_word_id[-1] == end_token):
            break

    return caption_word_id     