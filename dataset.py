# import some packages you need here
import os
import torch
from torch.utils.data import Dataset, DataLoader


class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):

        # write your codes here

        with open(input_file, 'r') as file:
            text = file.read()

        
        self.char2idx = {ch: idx for idx, ch in enumerate(sorted(set(text)))} # character to idx
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()} # idx to character


        self.text_indices = [self.char2idx[ch] for ch in text] # Convert the entire text into indices

        self.sequence_length = 30 # chunks

        self.num_sequences = len(self.text_indices) // self.sequence_length # num of sequences

    def __len__(self):
        return self.num_sequences

        # write your codes here

    def __getitem__(self, idx):

        # write your codes here
        start_idx = idx * self.sequence_length # chunk
        end_idx = start_idx + self.sequence_length

        input_seq = self.text_indices[start_idx:end_idx]
        target_seq = self.text_indices[start_idx + 1:end_idx + 1]

        # Convert to tensors
        input = torch.tensor(input_seq, dtype=torch.long)
        target = torch.tensor(target_seq, dtype=torch.long)

        return input, target

if __name__ == '__main__':

    # write test codes to verify your implementations

    
    input_file = 'shakespeare_train.txt'  

    dataset = Shakespeare(input_file)

    # Print the length of the dataset
    print(f"Number of sequences: {len(dataset)}")

    # Get a sample from the dataset
    sample_idx = 3
    input_sample, target_sample = dataset[sample_idx]

    # Print the input and target samples
    print("Input Sample:", input_sample)
    print("Target Sample:", target_sample)

    # Print the input and target samples as characters
    input_chars = ''.join([dataset.idx2char[idx.item()] for idx in input_sample])
    target_chars = ''.join([dataset.idx2char[idx.item()] for idx in target_sample])
    print("Input Sample (as chars):", input_chars)
    print("Target Sample (as chars):", target_chars)

    