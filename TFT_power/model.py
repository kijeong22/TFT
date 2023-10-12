import torch
import torch.nn as nn
from layer import (
    CategoricalEmbedding, 
    ContinuousEmbedding,
    VariableSelection,
    GRN,
    GLU,
    InterpretableMultiheadAttention,
    QuantileOutput
)

class TemporalFusionTransformer(nn.Module):
    def __init__(
            self, 
            encoder_len:int, 
            decoder_len:int, 
            d_model:int, 
            dropout:float,
            static_cate_num:list,
            static_conti_size:int,
            future_cate_num:list,
            category_num:list,
            continuous_input_size:int,
            static_num_input:int,
            encoder_num_input:int,
            decoder_num_input:int,
            num_heads:int,
            quantiles:list,
            tau:int,
            device
            ):
        
        super(TemporalFusionTransformer, self).__init__()

        self.q_sequence_len = decoder_len
        self.sequence_len = encoder_len + decoder_len
        self.dropout = dropout
        self.tau = tau

        # embedding
        self.static_cate_emb = CategoricalEmbedding(d_model, static_cate_num)
        self.static_conti_emb = ContinuousEmbedding(d_model, static_conti_size)
        self.future_emb = CategoricalEmbedding(d_model, future_cate_num)
        self.category_emb = CategoricalEmbedding(d_model, category_num)
        self.continuous_emb = ContinuousEmbedding(d_model, continuous_input_size)

        self.static_variable_selection = VariableSelection(d_model, static_num_input, dropout)
        self.encoder_variable_selection = VariableSelection(d_model, encoder_num_input, dropout)
        self.decoder_variable_selection = VariableSelection(d_model, decoder_num_input, dropout)

        # static covariate encoders
        self.grn_cs = GRN(d_model, d_model, d_model, dropout) # variable selection context
        self.grn_cc = GRN(d_model, d_model, d_model, dropout) # lstm initial cell state
        self.grn_ch = GRN(d_model, d_model, d_model, dropout) # lstm initial hidden state
        self.grn_ce = GRN(d_model, d_model, d_model, dropout) # static enrichment context

        # lstm encoder-decoder
        self.lstm_encoder = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.lstm_decoder = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)

        self.locality_glu = GLU(d_model, d_model)
        self.locality_layernorm = nn.LayerNorm(d_model)

        # static enrichment
        self.static_enrichment_grn = GRN(d_model, d_model, d_model, dropout)

        # temporal self-attention
        self.mask = torch.ones(self.q_sequence_len, self.sequence_len, dtype=torch.bool, device=device).triu(diagonal=1)
        self.imha = InterpretableMultiheadAttention(d_model, num_heads, dropout)
        self.attention_glu = GLU(d_model, d_model)
        self.attention_layernorm = nn.LayerNorm(d_model)

        # position-wise feed-forward
        self.position_wise_grn = GRN(d_model, d_model, d_model, dropout)
        self.position_wise_glu = GLU(d_model, d_model)
        self.position_wise_layernorm = nn.LayerNorm(d_model)

        # quantile output
        self.quantile_output = QuantileOutput(d_model, quantiles, tau)

    def forward(self, 
                static_cate_input,
                static_conti_input,
                future_input, 
                category_input, 
                continuous_input
                ):
        
        # embedding
        # shape : (batch_size, sequence_length, num_of_features, embedding_dim)        
        static_cate_input = self.static_cate_emb(static_cate_input.to(torch.int)) # (batch,1,static_cate_num,d_model)
        static_conti_input = self.static_conti_emb(static_conti_input.to(torch.float)) # (batch,1,static_conti_num,d_model)
        static_input = torch.cat([static_cate_input, static_conti_input], dim=-2)
        future_input = self.future_emb(future_input.to(torch.int)) # (batch,decoder_len,future_num,d_model)
        category_input = self.category_emb(category_input.to(torch.int)) # (batch,encoder_len,cate_input_num,d_model)
        continuous_input = self.continuous_emb(continuous_input.to(torch.float)) # (batch,encoder_len,conti_input_num,d_model)

        # static covariates encoders
        static_encoder_input = self.static_variable_selection(static_input) # (batch,1,d_model)
        c_s = self.grn_cs(static_encoder_input) # (batch,1,d_model)
        c_c = self.grn_cc(static_encoder_input) # (batch,1,d_model)
        c_h = self.grn_ch(static_encoder_input) # (batch,1,d_model)
        c_e = self.grn_ce(static_encoder_input) # (batch,1,d_model)
 
        # lstm encoder-decoder
        past_input = torch.cat([category_input,continuous_input], dim=2)
        encoder_input = self.encoder_variable_selection(past_input, c_s)
        decoder_input = self.decoder_variable_selection(future_input, c_s)
        resid_vs = torch.cat([encoder_input, decoder_input], dim=1)

        encoder_output, (e_c_c, e_c_h) = self.lstm_encoder(encoder_input, (c_c.permute(1,0,-1), c_h.permute(1,0,-1)))
        decoder_output, _ = self.lstm_decoder(decoder_input, (e_c_c, e_c_h))
        enc_dec_output = torch.cat([encoder_output, decoder_output], dim=1)

        enc_dec_glu_output = self.locality_glu(enc_dec_output)
        static_enrich_input = self.locality_layernorm(resid_vs + enc_dec_glu_output)

        # static enrichment
        query = self.static_enrichment_grn(static_enrich_input, c_e)
        query = query[:,-self.tau:] # only prediction
        key = self.static_enrichment_grn(static_enrich_input, c_e)
        value = self.static_enrichment_grn(static_enrich_input, c_e)

        # temporal self-attention
        imha_output, attention = self.imha(query, key, value, self.mask) # (batch,q_sequence,d_model)
        
        attention_glu_output = self.attention_glu(imha_output)
        position_wise_input = self.attention_layernorm(query + attention_glu_output)

        # position-wise feed-forward
        position_wise_output = self.position_wise_grn(position_wise_input)
        position_wise_output = self.position_wise_glu(position_wise_output)
        position_wise_output = self.position_wise_layernorm(static_enrich_input[:,-self.tau:] + position_wise_output)

        # quantile output
        quantile_output = self.quantile_output(position_wise_output)

        return quantile_output, attention
