import re
import pickle
import numpy as np

from tqdm import tqdm
from transformers import BertTokenizer

MAX_LEN = 50

MAGBERT_SENA_MAP = {
    "train": "train",
    "dev": "valid",
    "test": "test"
}

SENA_DATA_FORMAT = [
    'raw_text',
    'audio',
    'vision',
    'id',
    'text',
    'text_bert',
    'annotations',
    'classification_labels',
    'regression_labels'
]

COPY_SENA_DATA = [
    'raw_text',
    'id',
    'text',
    'text_bert',
    'annotations',
    'classification_labels',
    'regression_labels'
]

# Missing files are ['2WGyTLYerpo$_$44', '5W7Z1C_fDaE$_$24', 'BioHAh1qJAQ$_$30'] for split mag train
# Missing files are ['lXPQBPVc5Cw$_$30'] for split mag test
TRAIN_REMOVE = ['2WGyTLYerpo$_$44', '5W7Z1C_fDaE$_$24', 'BioHAh1qJAQ$_$30']
TEST_REMOVE = ['lXPQBPVc5Cw$_$30']


def init_data_dict(n_samples, old_dict, split):
    data_dict = {}
    if split == 'valid':
        data_dict['audio'] = np.zeros((n_samples, 50, 74))
        data_dict['vision'] = np.zeros((n_samples, 50, 47))
        for k in COPY_SENA_DATA:
            data_dict[k] = old_dict[k].copy()
    else:
        if split == 'train':
            removed_samples = len(TRAIN_REMOVE)
        else:
            removed_samples = len(TEST_REMOVE)
        data_dict['audio'] = np.zeros((n_samples - removed_samples, 50, 74))
        data_dict['vision'] = np.zeros((n_samples - removed_samples, 50, 47))
        for k in COPY_SENA_DATA:
            print(k)
            v = old_dict[k]
            if isinstance(v, list):
                print(len(v))
                data_dict[k] = []
            else:
                data_shape = list(v.shape)
                data_type = v.dtype
                data_shape[0] = data_shape[0] - removed_samples
                data_shape = tuple(data_shape)
                data_dict[k] = np.empty(data_shape, dtype=data_type)
                print(v.shape)

    return data_dict


def get_tokenized_idxs(bert_tokens):
    tokenized_ids = []
    ## very inefficient but since we run it only once we dont care
    for i, token in enumerate(bert_tokens):
        if token.startswith("##"):
            # tokenized word
            if tokenized_ids: # not empty
                last_token = tokenized_ids[-1]
                if (i-1) == last_token[1]:
                    # already in the list
                    last_token[1] = i
                else:
                    # new token
                    tokenized_ids.append([i-1, i])
            else:
                # the very first token
                tokenized_ids.append([i-1, i])

    for i, (s, e) in enumerate(tokenized_ids):
        tokenized_ids[i][1] = e - s

    return tokenized_ids


def extend_array(array, ext_list):
    """Takes an array and inserts copies of its own lines within the array.
    Also appends a zero vector at the start and at the end due to [CLS] and    
    [SEP] tokens.
    Args:
        array (np.array): [seq_len x features]
        ext_list (List): [(idx_to_repeat, repetitions)]

    Note: idx_to_repeat is the CS idx, i.e., 0,1,2,3 and does not require
    additions etc
    """
    d_feat = array.shape[1]
    zeros = np.zeros((1, d_feat))
    new_array = np.copy(array)
    new_array = np.concatenate((zeros, new_array), axis=0)
    # tmp_array = np.copy(new_array)
    # offset = 0
    for repeat_idx, repeats in ext_list:
        row = new_array[repeat_idx, :] # due to [CLS] offset
        repeat_row = np.repeat([row], repeats, axis=0)
        new_array = np.insert(
            new_array,
            repeat_idx + 1, #+ offset,
            repeat_row,
            axis=0
        )
        # offset += repeats
    new_array = truncate(new_array)
    new_array = np.concatenate((new_array, zeros), axis=0)
    return new_array


def get_sena_data(sena_dict, sena_idx):
    sena_text = sena_dict['raw_text'][sena_idx]
    sena_audio = sena_dict['audio'][sena_idx]
    sena_vision = sena_dict['vision'][sena_idx]
    sena_bert = sena_dict['text_bert'][sena_idx]

    return sena_text, sena_audio, sena_vision, sena_bert


def get_magbert_data(mag_data, mag_idx):
    mag_text = mag_data[mag_idx][0]
    mag_audio = mag_data[mag_idx][2]
    mag_vision = mag_data[mag_idx][1]

    return mag_text, mag_audio, mag_vision


def truncate(array):
    if array.shape[0] > MAX_LEN - 1:
        print(f"Truncating array with shape {array.shape} to append [SEP] token")
        new_array = array[: MAX_LEN - 1, :]
        return new_array
    else:
        return array


def pad_zeros(array):
    seq_len = array.shape[0]
    d_feat = array.shape[1]
    pad_len = MAX_LEN - seq_len
    if pad_len > 0:
        padding = np.zeros((pad_len, d_feat))
        array = np.concatenate((array, padding))
    return array


if __name__ == "__main__":
    magbert_data_path = "/data/efthygeo/mmsa/mag_bert/mosi.pkl"
    sena_data_path = "/data/efthygeo/mmsa/mosi/Processed/aligned_50.pkl"
    new_sena_data_path = \
        "/data/efthygeo/mmsa/mosi/Processed/aligned_50_magbert_av.pkl"

    with open(magbert_data_path, 'rb') as fd:
        magbert_data = pickle.load(fd)

    with open(sena_data_path, 'rb') as fd:
        sena_data = pickle.load(fd)

    new_sena_data = {}
    ###########################################################################
    ## MAG-BERT data, List for train/dev/test, Each list has tuples with 3
    ## data = sample[0], label = sample[1], id = sample[2]
    ## t, a, v = data[0], data[2], data[1]
    ###########################################################################

    for mag_split_name, sena_split_name in MAGBERT_SENA_MAP.items():
        mag_split = magbert_data[mag_split_name]
        sena_split = sena_data[sena_split_name]
        new_sena_split = init_data_dict(
                sena_split['raw_text'].shape[0],
                sena_split,
                sena_split_name,
            )

        # aligned mag-bert lists
        mag_ids = []
        mag_data = []
        mag_labels = []
        for s in mag_split:
            # s[0]: List --> see data above
            mag_data.append(s[0])
            # mag_ids: WKA50ygbEKI[0], convert to sena ids
            id = s[2]
            # labels
            mag_labels.append(s[1])
            # get the list id and increase it by 1
            m = re.search(r"\[([A-Za-z0-9_]+)\]", id)
            replace_str = m.group(0)
            fix_id = str(int(m.group(1)) + 1)
            new_id = id.replace(replace_str, "$_$" + fix_id)
            mag_ids.append(new_id)

        # dev_sena = new_sena_data['valid']
        # sena_ids: WKA50ygbEKI$_$20
        sena_ids = sena_split['id']
        # sena_ids = new_sena_split['id']

        print(f"There are {len(mag_ids)} in MAG data")
        print(f"There are {len(sena_ids)} in SENA data")

        missing_train = []
        for k in sena_ids:
            if k in mag_ids:
                pass
            else:
                print(f"We have found missing sample {k} mag")
                missing_train.append(k)
        print(f"Missing files are {missing_train} for split {mag_split_name} in MAG")

        # for k in missing_train:
        #     idx = sena_ids.index(k)
        #     t, a, v = get_magbert_data(mag_data, idx)
        #     print(f"Audio shape is {a.shape}")
        #     print(a)
        #     print(v)
        #     print(f"Vision shape is {v.shape}")
        #     print(f"Text is {t}")
        #     print(f"Label is {mag_labels[idx]}")

        # missing_train = []
        # for k in mag_ids:
        #     if k in sena_ids:
        #         pass
        #     else:
        #         print(f"We have found missing sample {k} from sena")
        #         missing_train.append(k)
        # print(f"Missing files are {missing_train} for split {mag_split_name} in SENA")

        # intialize tokenizer
        tok = BertTokenizer.from_pretrained("bert-base-uncased")
        not_found = 0
        print(f"Getting data for {mag_split_name} split of MOSI")
        for i, mag_s in tqdm(enumerate(mag_ids)):
            try:
                sena_idx = sena_ids.index(mag_s)
            except ValueError:
                print("That item does not exist")
                not_found += 1

            # sena data
            # sena_text, sena_audio, sena_vision, sena_bert = \
            #     get_sena_data(new_sena_split, sena_idx)

            # mag data
            mag_text, mag_audio, mag_vision = get_magbert_data(mag_data, i)

            # mag tokenization
            mag_tokenized = tok(" ".join(mag_text))
            mag_token_ids = mag_tokenized["input_ids"]
            # covert back to wordpieces to find tokenized words
            mag_tokens = tok.convert_ids_to_tokens(mag_token_ids)
            tokenized_ids = get_tokenized_idxs(mag_tokens)

            # extend audiovisual features in tokenized words
            new_mag_audio = extend_array(mag_audio, tokenized_ids)
            new_mag_vision = extend_array(mag_vision, tokenized_ids)

            # # truncate if necessary
            # new_mag_audio = truncate(mag_audio)
            # new_mag_vision = truncate(mag_vision)

            # pad w. zeros
            new_mag_audio = pad_zeros(new_mag_audio)
            new_mag_vision = pad_zeros(new_mag_vision)

            # import pdb; pdb.set_trace()
            # print(new_mag_audio.shape)
            # new_sena_split['audio'][sena_idx, :, :] = new_mag_audio 
            # new_sena_split['vision'][sena_idx, :, :] = new_mag_vision 
            new_sena_split['audio'][i, :, :] = new_mag_audio
            new_sena_split['vision'][i, :, :] = new_mag_vision
            for k in COPY_SENA_DATA:
                if isinstance(new_sena_split[k], list):
                    new_sena_split[k].append(sena_split[k][i])
                else:
                    new_sena_split[k][i] = sena_split[k][i]

            # print(f"Text is {sena_text} vs {mag_text}")
            # print(f"Audio shape is {sena_audio.shape} vs {new_mag_audio.shape}")
            # print(f"Vision shape is {sena_vision.shape} vs {new_mag_vision.shape}")
            # import pdb; pdb.set_trace()

        print(f"Finished extracting data for the {mag_split_name} split")
        new_sena_data[sena_split_name] = new_sena_split

    with open(new_sena_data_path, 'wb') as fd:
        pickle.dump(new_sena_data, fd)


    # for k in TRAIN_REMOVE:
    #     idx = new_sena_data['train']['id'].index(k)
    #     for k, v in new_sena_data['train'].items():
    #         if isinstance(v, list):
    #             del v[idx]
    #         else:
    #             v = np.delete(v, idx, axis=0)

    # for k, v in new_sena_data['train'].items():
    #     print(k)
    #     if isinstance(v, list):
    #         print(len(v))
    #     else:
    #         print(v.shape)




    # common = 0
    # not_common = []
    # for k in mag_ids:
    #     if k in sena_ids:
    #         print(f"Found common as {k}")
    #         common += 1
    #     else:
    #         not_common.append(k)

    # print(f"Total commons are: {common}")
    # print(f"Total dev length is MAG: {len(mag_ids)} and SENA: {len(sena_ids_new)}")
    # print(f"Not Common ids are {len(not_common)}")
    # print(f"Not Common ids are {not_common}")

    # not_common2 = []
    # for k in sena_ids_new:
    #     if k not in mag_ids:
    #         not_common2.append(k)

    # print(f"Not Common ids are {len(not_common2)}")
    # print(f"Not Common ids are {not_common2}")
    # DEV_MAG2SENA = {}
    # DEV_SENA2MAG = {}
    # for k,l in zip(not_common, not_common2):
    #     DEV_MAG2SENA[k] = l
    #     DEV_SENA2MAG[l] = k

    # print(DEV_MAG2SENA)
    # print(DEV_SENA2MAG)