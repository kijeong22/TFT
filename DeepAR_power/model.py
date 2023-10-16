import torch
import torch.nn as nn
import random

class CategoricalEmbedding(nn.Module):
    def __init__(self, embedding_dim, category_num:list):
        super(CategoricalEmbedding, self).__init__()

        self.embedding = nn.ModuleList([nn.Embedding(num, embedding_dim) for num in category_num])

    def forward(self, input):

        categorical_output = []

        for idx, emb in enumerate(self.embedding):
            output = emb(input[:, :, idx:idx+1]) # (batch_size,sequence_len,1,d_model)
            categorical_output.append(output)

        return torch.cat(categorical_output, dim=2) # (batch_size,sequence_len,num_feature,d_model)

class Encoder(nn.Module):
    def __init__(self, conti_size, d_model, embedding_dim, category_num:list, num_layers=2, dropout=0.1):
        super(Encoder, self).__init__()

        self.embedding = CategoricalEmbedding(embedding_dim, category_num)
        self.lstm = nn.LSTM(conti_size + embedding_dim*len(category_num), d_model, num_layers, dropout=dropout, batch_first=True)

    def forward(self, cate, conti):

        cate_embed = self.embedding(cate.to(torch.int))
        cate_embed = cate_embed.view(conti.shape[0], conti.shape[1], -1) # (batch, encoder_len, num_cate_features*embedding_dim)

        encoder_input = torch.cat([cate_embed, conti], dim=-1)

        _, (hidden, cell) = self.lstm(encoder_input.to(torch.float))

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, conti_size, d_model, embedding_dim, category_num:list, device,
                 num_layers=2, dropout=0.1, t_forcing=1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.output_size = conti_size
        self.t_forcing = t_forcing
        self.device = device

        self.embedding = CategoricalEmbedding(embedding_dim, category_num)
        self.lstm = nn.LSTM(conti_size + embedding_dim*len(category_num), self.d_model, self.num_layers, dropout=dropout, batch_first=True)

        self.mu_linear1 = nn.Linear(self.num_layers, 1)
        self.mu_linear2 = nn.Linear(self.d_model, self.output_size)

        self.sigma_linear1 = nn.Linear(self.num_layers, 1)
        self.sigma_linear2 = nn.Linear(self.d_model, self.output_size)
        self.softplus = nn.Softplus()

    def forward(self, past_conti, dec_conti, future, hidden, cell, train_mode):

        decoder_len = dec_conti.shape[1]
        batch = dec_conti.shape[0]

        mu_seq = torch.zeros(batch, decoder_len, self.output_size, device=self.device)
        sigma_seq = torch.zeros(batch, decoder_len, self.output_size, device=self.device)
        output_seq = torch.zeros(batch, decoder_len, self.output_size, device=self.device)

        future_embed = self.embedding(future)
        future_embed = future_embed.view(batch, decoder_len, -1) # (batch, decoder_len, num_future_features*embedding_dim)

        past_conti = past_conti[:,-2:-1,:]

        for t in range(decoder_len):

            decoder_input = torch.cat([past_conti, future_embed[:,t:t+1,:]], dim=-1).to(torch.float)
            
            _, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))

            mu = self.mu_linear1(hidden.view(batch, self.d_model, self.num_layers)) # (batch, d_model, 1)
            mu = self.mu_linear2(mu.squeeze(-1)) # (batch, conti_size)

            sigma = self.sigma_linear1(hidden.view(batch, self.d_model, self.num_layers)) # (batch, d_model, 1)
            sigma = self.sigma_linear2(sigma.squeeze(-1)) # (batch, conti_size)
            sigma = self.softplus(sigma)

            mu_seq[:,t,:] = mu
            sigma_seq[:,t,:] = sigma

            likelihood = torch.distributions.normal.Normal(mu, sigma)
            output = likelihood.sample()

            output_seq[:,t,:] = output
            past_conti = output.unsqueeze(1)

            if train_mode:

                if random.random() < self.t_forcing:
                    past_conti = dec_conti[:, t:t+1, :]

        return mu_seq, sigma_seq, output_seq
    
class DeepAR(nn.Module):
    def __init__(self, conti_size, d_model, embedding_dim, enc_category_num, dec_category_num,
                 device, num_layers, dropout, t_forcing):
        super(DeepAR, self).__init__()

        self.encoder = Encoder(conti_size, d_model, embedding_dim, enc_category_num, num_layers, dropout)
        self.decoder = Decoder(conti_size, d_model, embedding_dim, dec_category_num, device, 
                               num_layers, dropout, t_forcing)

    def forward(self, cate, enc_conti, dec_conti, future, train_mode=True):

        hidden, cell = self.encoder(cate, enc_conti)
        mu_seq, sigma_seq, output_seq = self.decoder(enc_conti, dec_conti, future, hidden, cell, train_mode)

        return mu_seq, sigma_seq, output_seq