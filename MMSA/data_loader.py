import os
import gc
import logging
import pickle
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from datasets import Dataset, load_from_disk, concatenate_datasets
# from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, Subset
from functools import partial
import tiktoken
from transformers import BertTokenizer
from torch.utils.data._utils.collate import default_collate


import matplotlib.pyplot as plt
# Setup the plotting style
plt.style.use('ggplot')

__all__ = ['MMDataLoader']

logger = logging.getLogger('MMSA')

CH_GPT2_PATHS = {
    "gpt2-chinese-cluecorpussmall": "uer/gpt2-chinese-cluecorpussmall",
    "gpt2-medium-chinese-cluecorpussmall": "uer/gpt2-medium-chinese-cluecorpussmall",
    "gpt2-large-chinese-cluecorpussmall": "uer/gpt2-large-chinese-cluecorpussmall",
    "gpt2-xlarge-chinese-cluecorpussmall": "uer/gpt2-xlarge-chinese-cluecorpussmall",
}


def align(x, max_len):
    """ adopted from MMSA.models.singleTask.BI_ENC.align function
    """
    # print(f"@ np has shape {x.shape}")
    # convert to torch.tensor and add virtual batch dimension
    x = torch.from_numpy(x).unsqueeze(0)
    
    # apply align function
    raw_seq_len = x.size(1)
    if raw_seq_len == max_len:
        # torch --> np
        x = x.squeeze(0)
        x = x.numpy()
        return x
    if raw_seq_len // max_len == raw_seq_len / max_len:
        pad_len = 0
        pool_size = raw_seq_len // max_len
    else:
        pad_len = max_len - raw_seq_len % max_len
        pool_size = raw_seq_len // max_len + 1
    pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
    x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, max_len, -1)
    x = x.mean(dim=1)
    
    # discard virtual batch dimension
    x = x.squeeze(0)
    # back to np
    x = x.numpy()
    return x


def pad_or_pool(array, max_len):
    """
    Pads a given array to max_len with zeros if len < max_len.
    Otherwise, it pools the array to the max_len by taking consecutive averages.

    Args:
        array (np.array): the input 2D array of size (D, T) that we need to pad or pool to max_len
        max_len (int): the desired maximum length of the return arg
    Outputs:
        (np.array): 2D array of size (D, max_len)
    """
    current_len, d = array.shape
    if current_len >= max_len:
        pp_array = align(array, max_len)
    else: # < max_len case
        # print(f"dimension before padding is {array.shape}")
        pp_array = np.pad(array, ((0, max_len - current_len), (0, 0)), 'constant')
        # print(f"dimension after padding is {pp_array.shape}")
    return pp_array, max_len


def pad_trim_or_pool(array, max_len, method='trim', pool=False):
        """
        Adjusts the length of an array by either padding, trimming, or pooling.

        Args:
        - array (np.array): Input array of size [L, D]
        - max_len (int): Desired maximum length of the array.
        - method (str): Method to adjust the length ('trim', 'pool').

        Returns:
        - np.array: The adjusted array.
        """
        current_len, d = array.shape
        # First adjust the length to max_len either by padding or trimming
        if current_len > max_len:
            adjusted = array[:max_len, :]
        else:
            adjusted = np.pad(array, ((0, max_len - current_len), (0, 0)), 'constant')
        # reshaped_data = adjusted.reshape(max_len//2, 2, d)
        # reshaped_data = reshaped_data.mean(axis=1)
        return adjusted, max_len 
        # return reshaped_data, max_len //2
        ###########################################################################################
        # # Now apply average pooling to halve the length
        # # This assumes the length of 'adjusted' is even. If it's odd, one extra element is ignored.
        # # total_len = len(adjusted)
        # if pool:
        # total_len = len(adjusted) // 2
        # print(adjusted.shape)
        # pooled = np.mean(adjusted.reshape(-1, 2), axis=1)
        # print(pooled.shape)
        
        ###########################################################################################
        

def get_seq_len(args, text_shape, a, v):
    if args.get('use_bert'):
        return (text_shape[2], a, v)
    else:
        return (text_shape[1], a, v)


def MM_hf_Dataset(args, mode='train'):
    use_augmentation = args.use_augmentation
    return_real_len = False
    if use_augmentation:
        return_real_len = \
            True if "fere" in args.augmentation.name else False

    if args.get("lm", None):
        print("Using GPT LM")
        use_gpt_lm = True
        # input: <eot> + sentence
        # target: setence + <eot>
        max_token_len = args.max_token_len #+ 1 # 50 + eot
        if "chinese" in args["lm"]:
            print(f"Using Chinese LM")
            chinese_lm = True
            enc = BertTokenizer.from_pretrained(CH_GPT2_PATHS[args["lm"]])
        else:
            chinese_lm = False
            enc = tiktoken.get_encoding("gpt2")
            if args.pad_token == "<|endoftext|>":
                pad_token = enc.eot_token
            else:
                pad_token = enc(args.pad_token)
        seq_lens = args["seq_lens"]
        print(
            f"All the sequence lengths are L:{seq_lens[0]}, A:{seq_lens[1]}, V:{seq_lens[2]}"
        )
        
    else:
        use_gpt_lm = False

    def get_real_len(example):
        real_len_mask = example['text_bert'][:, 1, :].astype(np.int)
        example['real_len'] = np.sum(real_len_mask, axis=1)
        return example

    def use_bert(use_embed, example):
        '''
        use embed (bool): when true means that we use the whole architecture
        when false we use just the embeddings
        '''
        if use_embed:
            # uses pretrained bert embeddings
            example['text'] = np.array(example['text']).astype(np.float32)
        else:
            # uses tokens and whole bert
            example['text'] = np.array(example['text_bert']).astype(np.float32)
        return example

    def av_preproc(example):
        example['audio'] = np.array(example['audio']).astype(np.float32)
        example['audio'][example['audio'] == -np.inf] = 0
        example['vision'] = np.array(example['vision']).astype(np.float32)
        return example

    def av_preproc_fs2(example):
        example['audio'], example['audio_lengths'] = pad_trim_or_pool(
            np.array(example['audio_full']).astype(np.float32),
            max_len=seq_lens[1], method='trim'
        )
        example['audio'][example['audio'] == -np.inf] = 0
        example['vision'], example['vision_lengths'] = pad_trim_or_pool(
            np.array(example['vision_full']).astype(np.float32),
            max_len=seq_lens[2], method='trim'
        )
        return example

    ##############################################################################################
    # TODO: remive in the future, DEAD CODE
    def idx_proc(example, idx):
        # import pdb; pdb.set_trace()
        example['index'] = np.array([idx]).astype(np.int64)
        example['id'] = np.array(example['id'])
        example['raw_text'] = np.array(example['raw_text'])
    ##############################################################################################

    def label_proc(example):
        example['labels'] = {
            "M": np.array(example['regression_labels']).astype(np.float32).reshape(-1)
        }
        return example

    def sims_labels(example):
        example['labels']['T'] = example['regression_labels_T'].astype(np.float32).reshape(-1)
        example['labels']['A'] = example['regression_labels_A'].astype(np.float32).reshape(-1)
        example['labels']['V'] = example['regression_labels_V'].astype(np.float32).reshape(-1)
        return example

    def normalize(example):
        example['vision'] = np.mean(example['vision'], axis=0, keepdims=True)
        example['audio'] = np.mean(example['audio'], axis=0, keepdims=True)
        # remove possible NaN values
        example['vision'][example['vision'] != example['vision']] = 0
        example['audio'][example['audio'] != example['audio']] = 0
        return example
    
    def ch_gpt_process(example):
        # print("entered here")
        tokens = enc(
            example['raw_text'], return_tensors="pt",
            padding="max_length", truncation=True,
            max_length=max_token_len+1 # should have been max_len + 1 but for fair comparisson
        ) # encode_ordinary ignores any special tokens
        tmp = tokens["input_ids"].reshape(-1)
        example["gpt_tokens_in"] = tmp[:max_token_len]
        example["gpt_tokens_tgt"] = tmp[1:] # end is maxlen
        example["gpt_tgt_mask"] = (tokens["attention_mask"].reshape(-1))[:max_token_len]
        return example

    def gpt_process(example):
        # print("entered here")
        token_ids = enc.encode_ordinary(example['raw_text']) # encode_ordinary ignores any special tokens
        real_token_len = len(token_ids)
        tgt_mask = np.ones(max_token_len, dtype=int)
        if real_token_len >= (max_token_len): 
            # trim to (max_len-1) and append <eot> in the start
            token_ids = token_ids[:max_token_len]
            # new len is max_token_len + 1
            token_ids = [pad_token] + token_ids
        elif real_token_len < (max_token_len):
            pad_len = max_token_len - real_token_len
            token_ids = [pad_token] + token_ids + [pad_token] * pad_len
            # mask_id = 0
            ignore_id = -1
            # ignore_mask = ignore_id * np.ones(pad_len)
            # import pdb; pdb.set_trace()
            tgt_mask[(real_token_len+1):] = ignore_id
        else:
            pass
        example["gpt_tokens_in"] = token_ids[:max_token_len] # up to original max_token_len in cgf
        example["gpt_tokens_tgt"] = token_ids[1:] # shifted one place
        example["gpt_tgt_mask"] = tgt_mask
        # import pdb; pdb.set_trace()
        return example

    def init_mosi():
        if args.get("hfPath", False):
            if args['model_name'] == 'mult':
                hf_data_path = os.path.join(args['hfPath'], f"hf_aligned_50_{mode}.arrow")
            else:
                # selfmm, misa, mms2s models
                hf_data_path = os.path.join(args['hfPath'], f"hf_unaligned_{max_token_len}_{mode}.arrow")
                # hf_data_path = os.path.join(args['hfPath'], f"hf_unaligned_39_{mode}.arrow")
            print(f"entered {hf_data_path} for mode {mode} and dataset {args['dataset_name']}")
            data = load_from_disk(hf_data_path)
        else:
            if args['custom_feature']:
                # use custom feature file extracted with MMSA-FET
                data_dict = pd.read_pickle(args['custom_feature'])
            else:
                print("Preprocessing custom M-SENA pickles")
                # use deault feature file specified in config file
                data_dict = pd.read_pickle(args['featurePath'])

            print(f"audio features are {data_dict['train']['audio'].shape}")
            print(f"vision features are {data_dict['train']['vision'].shape}")
            # import pdb; pdb.set_trace()
            # load as hf datasets and convert from dict of dicts to dict beacuse we know the split
            data = Dataset.from_dict(data_dict[mode])
            total_len = len(data)
            data = data.add_column("index", np.arange(total_len))

            print(f"Starting processing --------------------->")
            if use_gpt_lm:
                # import pdb; pdb.set_trace()
                # add gpt lm functionality
                if chinese_lm:
                    print("Using chinese causal LM")
                    data = data.map(ch_gpt_process)
                else:
                    # english language version
                    data = data.map(gpt_process)
            else:
                if return_real_len:
                    data = data.map(get_real_len)
                # new_features = data.features.copy()
                # PLM usage
                if args.get('use_bert', None):
                    use_bert_embed = partial(use_bert, False)
                    data = data.map(use_bert_embed, num_proc=16, batched=True)
                    # self.text = data[self.mode]['text_bert'].astype(np.float32)
                    args['feature_dims'][0] = 768
                else:
                    use_bert_token = partial(use_bert, True)
                    data = data.map(use_bert_token, num_proc=8, batched=True, batch_size=256)

            data = data.map(av_preproc, num_proc=8, batched=True, batch_size=256)
            data.set_format(type='numpy', columns=['text', 'audio', 'vision'], output_all_columns=True)

            data = data.map(label_proc)
            if args['dataset_name'] == 'sims':
                data = data.map(sims_labels)

            # TODO: remove the next line -- dead code
            data = data.map(idx_proc, with_indices=True)

            if args['dataset_name'] == 'sims':
                data = \
                    data.remove_columns(
                        ["text_bert",
                         "classification_labels",
                         "regression_labels"]
                    )
            else:
                data = \
                    data.remove_columns(
                        ["text_bert",
                         "annotations",
                         "classification_labels",
                         "regression_labels"]
                    )

            if not args['need_data_aligned']:
                if args['feature_A']:
                    #### dead code
                    # audio_lengths = list(data_A[self.mode]['audio_lengths'])
                    pass
                else:
                    audio_lengths = data['audio_lengths']
                if args['feature_V']:
                    #### dead code
                    # self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
                    pass
                else:
                    vision_lengths = data['vision_lengths']
                ## already in the dict no need to write a process function

            if args.get('need_normalized'):
                data = data.map(normalize)

            print(data["audio"].shape)
            print(data["vision"].shape)
            # import pdb; pdb.set_trace()
            if args.get('seq_len'):
                args['seq_len'] = get_seq_len(
                    args,
                    data["text"].shape,
                    data["audio"].shape[1],
                    data["vision"].shape[1]
                )
            # import pdb; pdb.set_trace()
            data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s/hf_unaligned_50_{mode}.arrow")
            # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s/hf_unaligned_39_{mode}.arrow")

        return args, data

    def init_mosei():
        return init_mosi()

    def init_sims():
        return init_mosi()

    def process_all_mode_datasets(args, mode):
        # List to hold all datasets loaded for the specified mode
        datasets = []
        
        # Walk through the directory and find all files related to the specified mode
        for root, dirs, files in os.walk(args['hfPath']):
            for f_dir in dirs:
                if f"{mode}" in f_dir:  # Ensures the mode is part of the filename just before the file extension
                    dir_path = os.path.join(root, f_dir)
                    print(f"Loading dataset from {dir_path}")
                    dataset = load_from_disk(dir_path)
                    datasets.append(dataset)
        # import pdb; pdb.set_trace()
        # Concatenate all loaded datasets
        if len(datasets) > 1:
            unified_dataset = concatenate_datasets(datasets, axis=0)
        else:
            unified_dataset = datasets[0]
        print("All datasets concatenated successfully.")
        # Clean up to free memory
        del datasets
        gc.collect()
        return unified_dataset
    
    def init_fs2():
        # if False:
        if args.get("hfPath", False):
            if args['model_name'] == 'mult':
                hf_data_path = os.path.join(args['hfPath'], f"hf_aligned_50_{mode}.arrow")
                data = load_from_disk(hf_data_path)
            else:
                # selfmm, misa, mms2s models
                # hf_data_path = os.path.join(args['hfPath'], f"hf_unaligned_50_{mode}.arrow")
                # hf_data_path = os.path.join(args['hfPath'], f"hf_unaligned_39_{mode}.arrow")
                # import pdb; pdb.set_trace()
                data = process_all_mode_datasets(args, mode)
            
            # if args["dataset_name"] == "mosi":
            #     data = data.remove_columns(["audio_full", "vision_full"])
            # print(f"entered {args['hfPath']} for mode {mode} and dataset {args['dataset_name']}")
            # import pdb; pdb.set_trace()
        else:
            print(f"Ongoing with Audio-Len = {seq_lens[1]}")
            print(f"Ongoing with Video-Len = {seq_lens[2]}")
            # path to features
            path_to_vision_feats = os.path.join(args["feature_path"], args["vision_feats"])
            path_to_audio_feats = os.path.join(args["feature_path"], args["audio_feats"])
            # Load the labels data
            if args["dataset_name"] == "sims":
                sims_headers = [
                    "video_id",
                    "clip_id",
                    "text",
                    "label",
                    "label_T",
                    "label_A",
                    "label_V",
                    "annotation",
                    "mode"
                ]
                labels_df = \
                    pd.read_csv(
                        os.path.join(
                            os.path.dirname(args["feature_path"]),
                            'label.csv'
                        ),
                        names=sims_headers,
                    )        
            else:
                labels_df = \
                    pd.read_csv(
                        os.path.join(
                            os.path.dirname(args["feature_path"]),
                            'label.csv'
                        )
                    )
            # Determine unique modes (train, valid, test)
            # modes = labels_df['mode'].unique()
           
            # Iterate through each row in the DataFrame
            # everytime loop over whole dataset and keep the corresponding split only!
            if args["dataset_name"] == "mosei" and args["fs2"]:
                if mode == 'train':
                    # splits for L = [50, 500, 700]
                    # number_of_splits = 8

                    # s-splits have L = [50, 100, 100]
                    number_of_splits = 4
                    global_df = labels_df[labels_df['mode'] == 'train']
                elif mode == 'test':
                    # number_of_splits = 2
                    number_of_splits = 1
                    global_df = labels_df[labels_df['mode'] == 'test']
                else: # valid
                    # number_of_splits = 1
                    number_of_splits = 1
                    global_df = labels_df[labels_df['mode'] == 'valid']
                total_length = len(global_df)
                global_df.drop(['label_T', 'label_A', 'label_V'], axis=1, inplace=True)
                split_lengths = [total_length // number_of_splits] * number_of_splits
                for i in range(total_length % number_of_splits):
                    split_lengths[i] += 1
                
                # to retrieve start and end indices
                split_indices_tmp = np.cumsum(split_lengths)
                split_indices = np.insert(split_indices_tmp, 0, 0)
                # split_flags[split_id] = True
                # import pdb; pdb.set_trace()            
            
                for split_id in range(number_of_splits):
                    # redefine bool flags to false at every loop
                    split_flags = [False] * number_of_splits
                    split_flags[split_id] = True
                    start_idx = split_indices[split_id]            
                    end_idx = split_indices[split_id+1]
                    iter_df = global_df.iloc[start_idx: end_idx]
                    # Prepare a dictionary to hold the max lengths
                    max_lengths = {'audio': 0, 'vision': 0}
                    # Prepare a dictionary to hold the lengths
                    lengths = {'audio': [], 'vision': []}
                    if args["dataset_name"] == "mosei":
                        # Define data dict
                        data_dict = {
                                "id": [],
                                "audio": [],
                                "vision": [],
                                "raw_text": [],
                                "audio_lengths": [],
                                "vision_lengths": [],
                                "annotations": [],
                                "regression_labels": [],
                        }
                    else:
                        # Define data dict
                        data_dict = {
                                "id": [],
                                "audio_full": [],
                                "vision_full": [],
                                "raw_text": [],
                                "annotations": [],
                                "regression_labels": [],
                        }
                    
                    # for index, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
                    for index, row in tqdm(iter_df.iterrows(), total=len(iter_df)):
                        video_id = row['video_id']
                        if args["dataset_name"] == "sims":
                            clip_id = f"{row['clip_id']:04}"
                        else:
                            clip_id = row['clip_id']
                        split = row['mode']
                        audio_feature_path = os.path.join(path_to_audio_feats, video_id, f"{clip_id}.npy")
                        vision_feature_path = os.path.join(path_to_vision_feats, video_id, f"{clip_id}.npy")

                        # Load features if they exist
                        if os.path.exists(audio_feature_path) and os.path.exists(vision_feature_path) and split == mode:
                            if args["dataset_name"] == "mosei":
                                # original implementation
                                # audio_features, audio_len = pad_trim_or_pool(
                                #     np.load(audio_feature_path).astype(np.float32),
                                #     max_len=seq_lens[1],
                                #     method='trim'
                                # )
                                # audio_features[audio_features == -np.inf] = 0
                                # vision_features, vision_len = pad_trim_or_pool(
                                #     np.load(vision_feature_path).astype(np.float32),
                                #     max_len=seq_lens[2],
                                #     method='trim'
                                # )
                                
                                # cheaper (post) implementation
                                audio_features, audio_len = pad_or_pool(
                                    np.load(audio_feature_path).astype(np.float32),
                                    max_len=seq_lens[1]
                                )
                                audio_features[audio_features == -np.inf] = 0
                                vision_features, vision_len = pad_or_pool(
                                    np.load(vision_feature_path).astype(np.float32),
                                    max_len=seq_lens[2]
                                )
                            else:
                                audio_features = np.load(audio_feature_path)
                                vision_features = np.load(vision_feature_path)
                            # Update maximum lengths
                            max_lengths['audio'] = \
                                max(max_lengths['audio'], len(audio_features))
                            max_lengths['vision'] = \
                                max(max_lengths['vision'], len(vision_features))
                            # Update lengths
                            if args["dataset_name"] == "mosei":
                                lengths['audio'].append(audio_len)
                                lengths['vision'].append(vision_len)
                            else:
                                lengths['audio'].append(len(audio_features))
                                lengths['vision'].append(len(vision_features))
                            # Update data dict
                            data_dict['id'].append(f"{video_id}_{clip_id}")
                            
                            if args["dataset_name"] == "mosei":
                                data_dict['audio'].append(audio_features)
                                data_dict['audio_lengths'].append(len(audio_features))
                                data_dict['vision'].append(vision_features)
                                data_dict['vision_lengths'].append(len(vision_features))
                            else:    
                                data_dict['audio_full'].append(audio_features)
                                # data_dict['audio_lengths'].append(len(audio_features))
                                data_dict['vision_full'].append(vision_features)
                                # data_dict['vision_lengths'].append(len(vision_features))
                            data_dict['raw_text'].append(row['text'])
                            data_dict['annotations'].append(row['annotation'])
                            data_dict['regression_labels'].append(row['label'])
                            # if args["dataset_name"] == "mosei" and args["use_splits"]:
                            #     cnt += 1
                            #     if cnt == split_1:
                            #         break
                    # import pdb; pdb.set_trace()

                    print(f"The maximum audio and vision lengths are ....")
                    print(max_lengths)
                    # Plotting, calculating stats, and saving the histograms
                    for modality, data in lengths.items():
                        len_data = len(data)
                        # for modality, len_data in data.items():
                        if len_data > 0:  # Ensure there is data to process
                            len_array = np.array(data)
                            mean_length = np.mean(len_array)
                            median_length = np.median(len_array)
                            max_length = np.max(len_array)

                            plt.figure(figsize=(10, 6))
                            plt.hist(len_array, bins=2000, color='steelblue', alpha=0.8)
                            # plt.hist(len_array, bins=300, color='steelblue', alpha=0.8)
                            plt.title(f'Length Distribution of {modality.capitalize()} Features in {mode.capitalize()} Split')
                            plt.xlabel('Length of Features')
                            plt.ylabel('Frequency')
                            plt.axvline(mean_length, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_length:.2f}')
                            plt.axvline(median_length, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_length:.2f}')
                            plt.axvline(max_length, color='black', linestyle='dashed', linewidth=1, label=f'Max: {max_length}')
                            plt.legend()
                            plt.grid(True)
                            filename = f'{mode}_{modality}_len.png'
                            plt.savefig(
                                os.path.join(args["feature_path"],filename),
                                dpi=300
                            )
                            plt.close()
                            print(f"{modality.capitalize()} features in {mode.capitalize()} split: Mean={mean_length:.2f}, Median={median_length:.2f}, Max={max_length}")

                    print("Distribution plots have been saved and statistics calculated.")
                    # load as hf datasets and convert from dict of dicts to dict beacuse we know the split
                    # define data dict
                    # with open(os.path.join(args['hfPath'], f'{mode}_hbrt.pickle'), 'wb') as handle:
                    #     pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)                
                    # import pdb; pdb.set_trace()
                    # import pdb; pdb.set_trace()

                    
                    data = Dataset.from_dict(
                        data_dict 
                    )
                    del data_dict
                
                    total_len = len(data)
                    data = data.add_column("index", np.arange(total_len))
                    # Printing the keys of the dataset, which are the column names
                    print(f"Starting processing --------------------->")
                    if use_gpt_lm:
                        # import pdb; pdb.set_trace()
                        # add gpt lm functionality
                        if chinese_lm:
                            print("Using chinese causal LM")
                            data = data.map(ch_gpt_process)
                        else:
                            print('Using English GPT2')
                            # english language version
                            # Issue says that we cannot use num_proc>1
                            data = data.map(gpt_process)
                    else:
                        print(f"Using Encoder LM --------------")
                        if return_real_len:
                            data = data.map(get_real_len)
                        # new_features = data.features.copy()
                        # PLM usage
                        if args.get('use_bert', None):
                            use_bert_embed = partial(use_bert, False)
                            data = data.map(use_bert_embed, num_proc=16, batched=True)
                            # self.text = data[self.mode]['text_bert'].astype(np.float32)
                            args['feature_dims'][0] = 768
                        else:
                            use_bert_token = partial(use_bert, True)
                            data = data.map(use_bert_token, num_proc=8, batched=True, batch_size=256)
                    
                    if args["dataset_name"] == "mosei":
                        print(f"AV features already in place")
                    else:
                        # trim & convert to np.32
                        # TODO: I dont know why but map can hang in large datasets
                        # one potenital fix is write_batch_size=2000. INvsetigate
                        # Its usage is set in the reading I/O phase for faster & non-buggy implementation!
                        data = data.map(av_preproc_fs2, num_proc=2, writer_batch_size=100)
                        # data = data.map(av_preproc_fs2, num_proc=8)
                        print(f"Removing columns")
                        data.remove_columns(["audio_full", "vision_full"])
                    data.set_format(
                        type='numpy',
                        columns=['raw_text', 'audio', 'vision'],
                        output_all_columns=True
                    )
                    # get common label notation
                    print(f"Fixing labels")
                    data = data.map(label_proc)
                    # if args['dataset_name'] == 'sims':
                    #     data = data.map(sims_labels)
                    # remove dead columns
                    if args['dataset_name'] == 'sims':
                        data = \
                            data.remove_columns(
                                [
                                    # "text_bert",
                                    # "classification_labels",
                                    "regression_labels"
                                ]
                            )
                    else:
                        data = \
                            data.remove_columns(
                                [
                                    # "text_bert",
                                    "annotations",
                                    # "classification_labels",
                                    "regression_labels"
                                ]
                            )

                    if args.get('need_normalized'):
                        data = data.map(normalize)

                    if args.get('seq_len'):
                        args['seq_len'] = get_seq_len(
                            args,
                            data["text"].shape,
                            data["audio"].shape[1],
                            data["vision"].shape[1]
                        )
                    print(f"Saving to disk")
                    # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_e2v_long_half/hf_unaligned_50_{mode}_s_{split_id}.arrow")
                    # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_hbrt_short/hf_unaligned_50_{mode}_s_{split_id}.arrow")
                    data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_e2v_short/hf_unaligned_50_{mode}_s_{split_id}.arrow")
                    # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_hbrt/hf_unaligned_50_{mode}_s_{split_id}.arrow")
                    # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_e2v_long_half/hf_unaligned_50_{mode}_s_{split_id}.arrow")
                    # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_hbrt_long_half/hf_unaligned_50_{mode}_s_{split_id}.arrow")

                    # del data_dict
                    del data
                    import gc
                    gc.collect()
                    data = []
            else:
                print(args['feature_dims'])
                print(args['seq_lens'])
                max_lengths = {'audio': 0, 'vision': 0}
                lengths = {'audio': [], 'vision': []}
                if args["dataset_name"] == "mosei":
                    # Define data dict
                    data_dict = {
                            "id": [],
                            "audio": [],
                            "vision": [],
                            "raw_text": [],
                            "audio_lengths": [],
                            "vision_lengths": [],
                            "annotations": [],
                            "regression_labels": [],
                    }
                else:
                    # Define data dict
                    data_dict = {
                            "id": [],
                            "audio_full": [],
                            "vision_full": [],
                            "raw_text": [],
                            "annotations": [],
                            "regression_labels": [],
                    }

                iter_df = labels_df[labels_df['mode'] == mode]
                for index, row in tqdm(iter_df.iterrows(), total=len(iter_df)):
                    video_id = row['video_id']
                    if args["dataset_name"] == "sims":
                        clip_id = f"{row['clip_id']:04}"
                    else:
                        clip_id = row['clip_id']
                    split = row['mode']
                    audio_feature_path = os.path.join(path_to_audio_feats, video_id, f"{clip_id}.npy")
                    vision_feature_path = os.path.join(path_to_vision_feats, video_id, f"{clip_id}.npy")

                    # Load features if they exist
                    if os.path.exists(audio_feature_path) and os.path.exists(vision_feature_path):
                        if args["dataset_name"] == "mosei":
                            audio_features, audio_len = pad_trim_or_pool(
                                np.load(audio_feature_path).astype(np.float32),
                                max_len=seq_lens[1],
                                method='trim'
                            )
                            audio_features[audio_features == -np.inf] = 0
                            vision_features, vision_len = pad_trim_or_pool(
                                np.load(vision_feature_path).astype(np.float32),
                                max_len=seq_lens[2],
                                method='trim'
                            )
                        else:
                            audio_features = np.load(audio_feature_path)
                            vision_features = np.load(vision_feature_path)
                            # import pdb; pdb.set_trace()
                        # Update maximum lengths
                        max_lengths['audio'] = \
                            max(max_lengths['audio'], len(audio_features))
                        max_lengths['vision'] = \
                            max(max_lengths['vision'], len(vision_features))
                        # Update lengths
                        if args["dataset_name"] == "mosei":
                            lengths['audio'].append(audio_len)
                            lengths['vision'].append(vision_len)
                        else:
                            lengths['audio'].append(len(audio_features))
                            lengths['vision'].append(len(vision_features))
                        # Update data dict
                        data_dict['id'].append(f"{video_id}_{clip_id}")
                        
                        if args["dataset_name"] == "mosei":
                            data_dict['audio'].append(audio_features)
                            data_dict['audio_lengths'].append(len(audio_features))
                            data_dict['vision'].append(vision_features)
                            data_dict['vision_lengths'].append(len(vision_features))
                        else:    
                            data_dict['audio_full'].append(audio_features)
                            # data_dict['audio_lengths'].append(len(audio_features))
                            data_dict['vision_full'].append(vision_features)
                            # data_dict['vision_lengths'].append(len(vision_features))
                        data_dict['raw_text'].append(row['text'])
                        data_dict['annotations'].append(row['annotation'])
                        data_dict['regression_labels'].append(row['label'])
                    data = Dataset.from_dict(
                    data_dict 
                )
                # # import pdb; pdb.set_trace()
                del data_dict
                # import pdb; pdb.set_trace()
                total_len = len(data)
                data = data.add_column("index", np.arange(total_len))
                # Printing the keys of the dataset, which are the column names

                print(f"Starting processing --------------------->")
                if use_gpt_lm:
                    # import pdb; pdb.set_trace()
                    # add gpt lm functionality
                    if chinese_lm:
                        print("Using chinese causal LM")
                        data = data.map(ch_gpt_process)
                    else:
                        print('Using English GPT2')
                        # english language version
                        # Issue says that we cannot use num_proc>1
                        data = data.map(gpt_process)
                else:
                    if return_real_len:
                        data = data.map(get_real_len)
                    # new_features = data.features.copy()
                    # PLM usage
                    if args.get('use_bert', None):
                        use_bert_embed = partial(use_bert, False)
                        data = data.map(use_bert_embed, num_proc=16, batched=True)
                        # self.text = data[self.mode]['text_bert'].astype(np.float32)
                        args['feature_dims'][0] = 768
                    else:
                        use_bert_token = partial(use_bert, True)
                        data = data.map(use_bert_token, num_proc=8, batched=True, batch_size=256)
                
                if args["dataset_name"] == "mosei":
                    print(f"AV features already in place")
                else:
                    # trim & convert to np.32
                    # TODO: I dont know why but map can hang in large datasets
                    # one potenital fix is write_batch_size=2000. INvsetigate
                    # Its usage is set in the reading I/O phase for faster & non-buggy implementation!
                    data = data.map(av_preproc_fs2, num_proc=2, writer_batch_size=100)
                    # data = data.map(av_preproc_fs2, num_proc=8)
                    print(f"Removing columns")
                    data = data.remove_columns(["audio_full", "vision_full"])
                data.set_format(
                    type='numpy',
                    columns=['raw_text', 'audio', 'vision'],
                    output_all_columns=True
                )
                # get common label notation
                print(f"Fixing labels")
                data = data.map(label_proc)
                # if args['dataset_name'] == 'sims':
                #     data = data.map(sims_labels)
                # remove dead columns
                if args['dataset_name'] == 'sims':
                    data = \
                        data.remove_columns(
                            [
                                # "text_bert",
                                # "classification_labels",
                                "regression_labels"
                            ]
                        )
                else:
                    data = \
                        data.remove_columns(
                            [
                                # "text_bert",
                                "annotations",
                                # "classification_labels",
                                "regression_labels"
                            ]
                        )

                if args.get('need_normalized'):
                    data = data.map(normalize)

                if args.get('seq_len'):
                    args['seq_len'] = get_seq_len(
                        args,
                        data["text"].shape,
                        data["audio"].shape[1],
                        data["vision"].shape[1]
                    )
                print(f"Saving to disk")
                
                # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_hbrt/hf_unaligned_50_{mode}_s_0")
                data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_e2v/hf_unaligned_50_{mode}_s_0")
                # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_hbrt_long/hf_unaligned_50_{mode}_s_0")
                # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_hbrt/hf_unaligned_50_{mode}.arrow")
                # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_hbrt_long/hf_unaligned_50_{mode}.arrow")
                # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_hbrt_long/hf_unaligned_50_{mode}.arrow")
                # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_e2v/hf_unaligned_50_{mode}.arrow")
                # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s_e2v_long/hf_unaligned_50_{mode}.arrow")
                # data.save_to_disk(f"/data/efthygeo/mmsa/{args['dataset_name']}/Processed/mms2s/hf_unaligned_50_{mode}.arrow")
        return args, data

    DATASET_MAP = {
            'mosi': init_mosi,
            'mosei': init_mosei,
            'sims': init_sims,
            'mosi_fs2': init_fs2,
            'mosei_fs2': init_fs2,
            'sims_fs2': init_fs2,
        }

    # use fs1 or fs2
    if args.get('fs2', None):
        suffix = "_fs2"
    else:
        suffix = ""

    return DATASET_MAP[args['dataset_name']+suffix]()


class MMDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        # return real len functionality
        self.use_augmentation = args.use_augmentation
        self.return_real_len = False
        if self.use_augmentation:
            self.return_real_len = \
                True if "fere" in args.augmentation.name else False
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
        }
        DATASET_MAP[args['dataset_name']]()

        
        # real_token_len = len(token_ids)
        # tgt_mask = np.ones(max_token_len, dtype=int)
        # if real_token_len >= (max_token_len): 
        #     # trim to (max_len-1) and append <eot> in the start
        #     token_ids = token_ids[:max_token_len]
        #     # new len is max_token_len + 1
        #     token_ids = [pad_token] + token_ids
        # elif real_token_len < (max_token_len):
        #     pad_len = max_token_len - real_token_len
        #     token_ids = [pad_token] + token_ids + [pad_token] * pad_len
        #     # mask_id = 0
        #     ignore_id = -1
        #     # ignore_mask = ignore_id * np.ones(pad_len)
        #     # import pdb; pdb.set_trace()
        #     tgt_mask[(real_token_len+1):] = ignore_id
        # else:
        #     pass
        # example["gpt_tokens_in"] = token_ids[:max_token_len] # up to original max_token_len in cgf
        # example["gpt_tokens_tgt"] = token_ids[1:] # shifted one place
        # example["gpt_tgt_mask"] = tgt_mask
        # import pdb; pdb.set_trace()
        # return example

    def __init_mosi(self):
        if self.args['custom_feature']:
            # use custom feature file extracted with MMSA-FET
            with open(self.args['custom_feature'], 'rb') as f:
                data = pickle.load(f)
        else:
            # use deault feature file specified in config file
            with open(self.args['featurePath'], 'rb') as f:
                data = pickle.load(f)

        # Get real len of sequence to apply augmentation on the real seq
        if self.return_real_len:
            real_len_mask = \
                data[self.mode]['text_bert'][:, 1, :].astype(np.int)
            self.real_len = np.sum(real_len_mask, axis=1)
        # PLM usage
        if self.args.get('use_bert', None):
            self.text = data[self.mode]['text_bert'].astype(np.float32)
            self.args['feature_dims'][0] = 768
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
            self.args['feature_dims'][0] = self.text.shape[2]
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.args['feature_dims'][1] = self.audio.shape[2]
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.args['feature_dims'][2] = self.vision.shape[2]
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        # Overide with custom modality features
        if self.args['feature_T']:
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            if self.args.get('use_bert', None):
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2]
        if self.args['feature_A']:
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2]
        if self.args['feature_V']:
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]

        self.labels = {
            # 'M': data[self.mode][self.args['train_mode']+'_labels'].astype(np.float32)
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }
        if self.args['dataset_name'] == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode]['regression' + '_labels_' + m].astype(np.float32)

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args['need_data_aligned']:
            if self.args['feature_A']:
                self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
            else:
                self.audio_lengths = data[self.mode]['audio_lengths']
            if self.args['feature_V']:
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            else:
                self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if self.args.get('data_missing'):
            # Currently only support unaligned data missing.
            self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:,0,:], self.text[:,1,:], None,
                                                                                        self.args.missing_rate[0], self.args.missing_seed[0], mode='text')
            Input_ids_m = np.expand_dims(self.text_m, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(self.text[:,2,:], 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)

            if self.args['need_data_aligned']:
                self.audio_lengths = np.sum(self.text[:,1,:], axis=1, dtype=np.int32)
                self.vision_lengths = np.sum(self.text[:,1,:], axis=1, dtype=np.int32)

            self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio, None, self.audio_lengths,
                                                                                        self.args.missing_rate[1], self.args.missing_seed[1], mode='audio')
            self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision, None, self.vision_lengths,
                                                                                        self.args.missing_rate[2], self.args.missing_seed[2], mode='vision')

        if self.args.get('need_normalized'):
            self.__normalize()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):

        if mode == 'text':
            input_len = np.argmin(input_mask, axis=1)
        elif mode == 'audio' or mode == 'vision':
            input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])
        np.random.seed(missing_seed)
        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask

        assert missing_mask.shape == input_mask.shape

        if mode == 'text':
            # CLS SEG Token unchanged.
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1

            modality_m = missing_mask * modality + (100 * np.ones_like(modality)) * (input_mask - missing_mask) # UNK token: 100.
        elif mode == 'audio' or mode == 'vision':
            modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality

        return modality_m, input_len, input_mask, missing_mask

    def __truncate(self):
        # NOTE: truncate input to specific length.
        def do_truncate(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature

        text_length, audio_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # For visual and audio modality, we average across time:
        # The original data has shape (max_len, num_examples, feature_dim)
        # After averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }
        if self.return_real_len:
            # add real_len tensor
            sample['real_len'] = self.real_len[index]
        if not self.args['need_data_aligned']:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        if self.args.get('data_missing'):
            sample['text_m'] = torch.Tensor(self.text_m[index])
            sample['text_missing_mask'] = torch.Tensor(self.text_missing_mask[index])
            sample['audio_m'] = torch.Tensor(self.audio_m[index])
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['audio_mask'] = self.audio_mask[index]
            sample['audio_missing_mask'] = torch.Tensor(self.audio_missing_mask[index])
            sample['vision_m'] = torch.Tensor(self.vision_m[index])
            sample['vision_lengths'] = self.vision_lengths[index]
            sample['vision_mask'] = self.vision_mask[index]
            sample['vision_missing_mask'] = torch.Tensor(self.vision_missing_mask[index])

        return sample


def load_indices_from_file(filename):
    indices = []
    with open(filename, 'r') as file:
        for line in file:
            index = int(line.strip())
            indices.append(index)
    return indices


def MMDataLoader(args, num_workers):
    if ('tetfn' in args['model_name']):
    # if ('sims' in args['dataset_name']) and ((args.get('lm', None) is None) and (args.get('fs2', False) == False)):
        datasets = {
            'train': MMDataset(args, mode='train'),
            'valid': MMDataset(args, mode='valid'),
            'test': MMDataset(args, mode='test'),
        }
        # import pdb; pdb.set_trace()
        if 'seq_lens' in args:
            args['seq_lens'] = datasets['train'].get_seq_len()
            # args['seq_lens'] = datasets['train'].get_seq_len()
    else:
        print("Loading HF datasets")
        datasets = {}
        print("---------------------- Ongoing with TRAIN data split -----------------------------")
        _, datasets['train'] = MM_hf_Dataset(args, mode='train')
        print("---------------------- Ongoing with VALID data split -----------------------------")
        _, datasets['valid'] = MM_hf_Dataset(args, mode='valid')
        print("---------------------- Ongoing with TEST data split -----------------------------")
        _, datasets['test'] = MM_hf_Dataset(args, mode='test')
        
        # for mosei case
        if 'use_subset' in args:
            subset_ids = \
                load_indices_from_file(f"MMSA/subsets/mosei/subset_{args['use_subset']}_indices.txt")
            train_subset = Subset(datasets['train'], subset_ids)
            datasets['train'] = train_subset


    # train_data = datasets['train']
    print(f"Ongoing with num_workers={num_workers}")
    collate_fn = None
    # COMMENT: minor fix for compatibility
    llm = args.get('lm', "None")
    use_custom_collator = args.get("use_custom_collator", False)
    if 'SmolLM' in llm or 'gpt' in llm or use_custom_collator:
        collate_fn = custom_collate_fn
    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True,
                       collate_fn=collate_fn,
                       )
        for ds in datasets.keys()
    }

    return dataLoader

###################################################################################################
# Custom collator function to resolve writable compatibility with torch 2.0
###################################################################################################
import copy
def ensure_writable(data):
    """
    Recursively ensure all elements in the data structure are writable and converted to tensors if needed.
    """
    if isinstance(data, np.ndarray):
        # Ensure NumPy array is writable, then convert to PyTorch tensor
        return torch.from_numpy(copy.deepcopy(data))  # Always copy to ensure writability
    
    elif isinstance(data, torch.Tensor):
        # Clone tensors to ensure they are writable
        return data.clone()
    
    elif isinstance(data, dict):
        # Recursively handle dictionaries
        return {k: ensure_writable(v) for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        # Recursively handle lists or tuples
        return type(data)(ensure_writable(v) for v in data)
    
    # Return other types unchanged
    return data

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches containing non-writable NumPy arrays.
    """
    # Apply `ensure_writable` to every element in the batch
    processed_batch = [ensure_writable(s) for s in batch]
    # Use PyTorch's default_collate to combine the processed batch into a final batch
    return default_collate(processed_batch)
###################################################################################################

# def MMDataLoader(args, num_workers):

#     datasets = {
#         'train': MMDataset(args, mode='train'),
#         'valid': MMDataset(args, mode='valid'),
#         'test': MMDataset(args, mode='test')
#     }

#     if 'seq_lens' in args:
#         args['seq_lens'] = datasets['train'].get_seq_len()
#         # args['seq_lens'] = datasets['train'].get_seq_len()

#     dataLoader = {
#         ds: DataLoader(datasets[ds],
#                        batch_size=args['batch_size'],
#                        num_workers=num_workers,
#                        shuffle=True)
#         for ds in datasets.keys()
#     }

#     return dataLoader
