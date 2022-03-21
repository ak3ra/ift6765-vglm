import copy
import os
import h5py
import torch
from torch.utils.data import  Dataset
import torch.nn.functional as F


class CoLDataset(Dataset):
    IGNORE_ID = -100
    sent_strategy = 'first'

    def __init__(self, file_path, tokenizer_name, tokenizer, block_size=126,
                 split_sent=False, voken_dir=None, suffix=None, verbose=False,
                 voken_ablation=None):

        # Open token's hdf5
        token_path = file_path + '.' + tokenizer_name + '.hdf5'
        assert os.path.isfile(token_path)
        if verbose:
            print("-------- Load Data -------")
            print("Load tokens from", token_path)
        self.token_hdf5 = h5py.File(token_path, 'r')
        self.tokenizer = tokenizer
        self.tokens = self.token_hdf5['tokens']
        self.verbose = verbose
        self.voken_ablation = voken_ablation
        self._iter_cnt = 0

        # Open voken's hdf5 and load voken ids

        self.vokens = None

        # Split for every block_size tokens
        # The last block without full length will be dropped.
        num_tokens = len(self.tokens)
        self.starts = list(range(0, num_tokens, block_size))
        self.batches = list(zip(self.starts[:-1], self.starts[1:]))


        if self.voken_ablation == 'token':
            self._voken_ids = list(range(30522))

    @property
    def voken_size(self):
        return len(self._voken_ids)

    @property
    def voken_ids(self):
        return copy.copy(self._voken_ids)

    def assert_equal_vokens(self, dataset):
        assert self.voken_size == dataset.voken_size
        for vid, vid1 in zip(self.voken_ids, dataset.voken_ids):
            assert vid == vid1

    def __len__(self):
        return len(self.batches) - 1

    def __getitem__(self, item):
        token_start, token_end = self.batches[item]
        tokens = list(self.tokens[token_start: token_end])
        token_tensor = torch.tensor(
            self.tokenizer.build_inputs_with_special_tokens(tokens),
            dtype=torch.long) # This might be problematic. ignore and CLS, and SEP tokens.
 
        return token_tensor
