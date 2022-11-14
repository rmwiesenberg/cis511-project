from collections import Counter
from torch import nn

from common import *
from turk_dataset import TurkDataset


class Homebrew(nn.Module):
    def __init__(
            self, *, vocab_size, output_size,
            embedding_dim, hidden_dim, n_layers
    ):
        super(Homebrew, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=0.5, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

        pass

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if TRAIN_ON_GPU:
            hidden = (weight.new(self.n_layers, batch_size,
                                 self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size,
                                 self.hidden_dim).zero_().cuda())
        else:
            hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


class HomebrewMetric(SimplicityMetric):
    """Computes an example simplicity metric as length_diff / norm_length"""
    def __init__(self, tune_data: SimplicityDataset):
        # This metric is not learned so no need to use tune_data.
        # But I include it for copying this example file.
        pass

    def __call__(self, simplicity_pair: SimplicityPair) -> float:
        """Return a simplicity value based on the norm and simp of the pair"""
        diff = len(simplicity_pair.norm) - len(simplicity_pair.simp)
        return diff / len(simplicity_pair.norm)


def main():
    tune_data = TurkDataset.from_tune()

    test_data = TurkDataset.from_test()
    compute_metric_main(HomebrewMetric(tune_data), test_data)


if __name__ == '__main__':
    main()
