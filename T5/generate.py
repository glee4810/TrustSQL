import time

import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler


def generate_sql(model, eval_dataset, args, collator, verbose=0):

    file_name = args.config.split('/')[-1]
    start_time = time.time()
    eval_sampler = SequentialSampler(eval_dataset)
    dataloader = DataLoader(
                                eval_dataset, 
                                sampler=eval_sampler, 
                                batch_size=args.eval_batch_size, 
                                drop_last=False,
                                collate_fn=collator
                                )
    tokenizer = eval_dataset.tokenizer
    model.eval()

    with torch.no_grad():

        out_eval = {}
        for batch_idx, batch in enumerate(dataloader, 1):

            input_ids = batch['inputs'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            db_ids = batch['db_id']
            data_ids = batch['id']
            generation_output = model.generate(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_length=args.max_length,
                                    num_beams=args.num_beams,
                                    return_dict_in_generate=True,
                                    output_scores=True
                                    )

            preds = generation_output['sequences'].cpu() if args.device == 'cuda' else generation_output['sequences']
            logits = torch.stack(generation_output['scores'], dim=1)[::int(args.num_beams)]
            logits = logits.cpu() if args.device == 'cuda' else logits
            probs = torch.softmax(logits, dim=2).float()
            log_probs = torch.log_softmax(logits, dim=2).float()
            max_prob, _ = torch.max(probs, axis=2)
            max_prob = max_prob.numpy()
            entropies = ( torch.sum(probs * log_probs, axis=2) * (-1) ).numpy()

            pred_list = []
            maxprob_list = []
            entropy_list = []
            for i in range(len(preds)):
                pred = tokenizer.decode(preds[i], skip_special_tokens=True)
                pred_tensor = preds[i][1:]
                maxprob_truncated = max_prob[i].tolist()
                entropy_truncated = entropies[i].tolist()
                if tokenizer.eos_token_id in pred_tensor:
                    pred_eos_idx = torch.nonzero(pred_tensor==tokenizer.eos_token_id)[0].item()
                    maxprob_truncated = maxprob_truncated[:pred_eos_idx+1]
                    entropy_truncated = entropy_truncated[:pred_eos_idx+1]
                pred_list.append(pred)
                maxprob_list.append(maxprob_truncated)
                entropy_list.append(entropy_truncated)

                result = {}
                result['db_id'] = db_ids[i]
                result['question'] = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                result['real'] = tokenizer.decode(labels[i], skip_special_tokens=True)
                result['pred'] = pred_list[i]
                result['maxprob'] = maxprob_list[i]
                result['entropy'] = entropy_list[i]
                out_eval[data_ids[i]] = result

            if verbose>0:
                print(f'{batch_idx}/{len(dataloader)} ({round(batch_idx/len(dataloader)*100, 4)}%) --- {file_name}', end='\r')

    if verbose>0:
        print(f"inference took {round(time.time() - start_time, 6)} secs")

    return out_eval