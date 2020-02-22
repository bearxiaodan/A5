#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from cnn import CNN
from highway import Highway

dropout_rate=0.3
charembedsize=50
m_word=21


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)



# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size
        self.vocab = vocab
        pad_token_idx = vocab.char2id['<pad>']
        self.charembed = nn.Embedding(len(vocab.char2id), charembedsize, padding_idx=pad_token_idx)
        self.cnn = CNN(charembedsize, m_word, embed_size)
        self.highway = Highway(embed_size)
        self.dropout = nn.Dropout(p=dropout_rate)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        xwordembed = []
        for word in input:
            xcharembed = self.charembed(word)
            xreshaped = xcharembed.permute(0, 2, 1)
            xconvout = self.cnn(xreshaped)
            xhighway = self.highway(xconvout)
            xwordembed.append(self.dropout(xhighway))
        output = torch.stack(xwordembed)
        return output

        ### END YOUR CODE
