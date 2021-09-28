import torch
import torch.nn as nn
import numpy as np


class PromptEncoder(torch.nn.Module):
    def __init__(self, hidden_size, prompt_length, prompt_encoder_head_type, vocab_size, device, input_embeddings=None):
        super(PromptEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.prompt_length = prompt_length
        self.prompt_encoder_head_type = prompt_encoder_head_type
        self.vocab_size = vocab_size + 1
        self.device = device

        self.prompt_embeddings = torch.nn.Embedding(self.prompt_length, self.hidden_size)

        if self.prompt_encoder_head_type == "raw":
            init_tokens = torch.from_numpy(np.random.choice(self.vocab_size, self.prompt_length))
            init_weight = input_embeddings(init_tokens)
            self.prompt_embeddings.weight.data.copy_(init_weight)

        elif self.prompt_encoder_head_type == "lstm":
            self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                           hidden_size=self.hidden_size,
                                           num_layers=2,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size, self.hidden_size))
        elif self.prompt_encoder_head_type == "mlp":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.hidden_size))
        else:
            raise NotImplementedError()

    def forward(self):
        input_ids = torch.LongTensor(list(range(self.prompt_length))).to(self.device)
        replace_embeds = self.prompt_embeddings(input_ids)

        if self.prompt_encoder_head_type == "raw":
            return replace_embeds
        elif self.prompt_encoder_head_type == "lstm":
            replace_embeds = replace_embeds.unsqueeze(0)        # [batch_size, prompt_length, embed_size]
            replace_embeds = self.lstm_head(replace_embeds)[0]  # [batch_size, prompt_length, 2 * hidden_size]
            if self.prompt_length == 1:
                replace_embeds = self.mlp_head(replace_embeds)
            else:
                replace_embeds = self.mlp_head(replace_embeds).squeeze()
            return replace_embeds
        elif self.prompt_encoder_head_type == "mlp":
            replace_embeds = self.mlp(replace_embeds)
            return replace_embeds
        else:
            raise NotImplementedError()