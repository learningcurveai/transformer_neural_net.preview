import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data
import math
import torch.nn.functional as F

class PositionEncodedEmbeddings(nn.Module):
    def __init__(self, vocab_size, model_dim, max_len = max_sentence_length+1):
        super(PositionEncodedEmbeddings, self).__init__()
        self.model_dim = model_dim
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.position_encoded_mat = self.create_positinal_encoding(max_len, model_dim)
        
    def create_positinal_encoding(self, max_len, model_dim):
        position_encoded_mat = torch.zeros(max_len, model_dim).to(device)
        # for each position of the word
        for pos in range(max_len):
            # for each dimension of the each position
            for i in range(0, model_dim):
                if (i % 2)==1:
                    position_encoded_mat[pos, i] = math.cos(pos / (10000 ** ((2 * (i))/model_dim)))
                else:
                    position_encoded_mat[pos, i] = math.sin(pos / (10000 ** ((2 * (i))/model_dim)))
        position_encoded_mat = position_encoded_mat.unsqueeze(0)
        return position_encoded_mat
        
    def forward(self, encoded_words):
        embedding = self.embed(encoded_words) * math.sqrt(self.model_dim)
        embedding += self.position_encoded_mat[:, :embedding.size(1)]
        return embedding
        
class MultiHeadAttention(nn.Module):
    
    def __init__(self, heads, d_model):
        
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0
        self.head_dim = d_model // heads
        self.heads = heads
        
        self.projection1 = nn.Linear(d_model, d_model) # a.k.a. the query matrix
        self.projection2 = nn.Linear(d_model, d_model) # a.k.a. the key matrix
        self.projection3 = nn.Linear(d_model, d_model) # a.k.a. the value matrix
        
        self.interaction = nn.Linear(d_model, d_model)
        
    def forward(self, projection1, projection2, projection3, mask):
        
        # (batch_size, max_len, d_model)
        projection1 = self.projection1(projection1)
        projection2 = self.projection2(projection2)        
        projection3 = self.projection3(projection3)
        
        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, head_dim) --> (batch_size, h, max_len, head_dim)
        projection1 = projection1.view(projection1.shape[0], -1, self.heads, self.head_dim).permute(0, 2, 1, 3)   
        projection2 = projection2.view(projection2.shape[0], -1, self.heads, self.head_dim).permute(0, 2, 1, 3)  
        projection3 = projection3.view(projection3.shape[0], -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        
        # (batch_size, h, max_len, head_dim) matmul (batch_size, h, head_dim, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(projection1, projection2.permute(0,1,3,2)) / math.sqrt(projection1.size(-1))
        
        # mask shape: (batch_size, 1, 1, max_len)
        scores = scores.masked_fill(mask == 0, -1e9)    # (batch_size, h, max_len, max_len)
        scores = F.softmax(scores, dim = -1)            # (batch_size, h, max_len, max_len)
        
        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, head_dim) --> (batch_size, h, max_len, head_dim)
        encoded_mat = torch.matmul(scores, projection3)
        
        # (batch_size, h, max_len, head_dim) --> (batch_size, max_len, h, head_dim) --> (batch_size, max_len, h * head_dim)
        encoded_mat = encoded_mat.permute(0,2,1,3).contiguous().view(encoded_mat.shape[0], -1, self.heads * self.head_dim)
        
        # (batch_size, max_len, h * head_dim)
        interaction = self.interaction(encoded_mat)
        
        return interaction
    
class FeedForward(nn.Module):

    def __init__(self, d_model, middle_dim = fully_connected_middle_dim):
        super(FeedForward, self).__init__()
        
        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out
    
class Encoder(nn.Module):

    def __init__(self, d_model, heads):
        super(Encoder, self).__init__()
        
        # To create encoded question
        self.self_attention = MultiHeadAttention(heads, d_model)
        
        self.feed_forward = FeedForward(d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.05)

    def forward(self, question_embeddings, question_mask):
    
        # Process question embedding
        interaction = self.self_attention(question_embeddings, question_embeddings, question_embeddings, question_mask)
        interaction = self.layernorm(interaction + question_embeddings)
        
        # Create encoded question
        feed_forward_out = self.dropout(self.feed_forward(interaction))
        encoded_question = self.layernorm(feed_forward_out + interaction)
        
        return encoded_question
        
class Decoder(nn.Module):
    
    def __init__(self, d_model, heads):
        super(Decoder, self).__init__()
        
        # To create encoded answer
        self.self_attention = MultiHeadAttention(heads, d_model)
        
        # To process encoded answer and encoded question
        self.multihead_attention = MultiHeadAttention(heads, d_model)
        
        self.feed_forward = FeedForward(d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.05)
        
    def forward(self, answer_embeddings, answer_mask, encoded_question, question_mask ):
            
        # Process answer embedding and create encoded answer
        interaction = self.self_attention(answer_embeddings, answer_embeddings, answer_embeddings, answer_mask)
        encoded_answer = self.layernorm(interaction + answer_embeddings)
        
        # Process encoded question and encoded answer
        interaction = self.multihead_attention(encoded_answer, encoded_question, encoded_question, question_mask)
        interaction = self.layernorm(interaction + encoded_answer)
        
        # Produce final output
        feed_forward_out = self.dropout(self.feed_forward(interaction))
        decoded_output = self.layernorm(feed_forward_out + interaction)
        
        return decoded_output
    
class Transformer(nn.Module):
    
    def __init__(self, d_model, heads, dictionary):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = len(dictionary)
        
        self.embed = PositionEncodedEmbeddings(self.vocab_size, d_model)
        self.encoder = Encoder(d_model, heads)
        self.decoder = Decoder(d_model, heads)
        
        self.sequence_generator = nn.Linear(d_model, self.vocab_size)
        
    def encode(self, question, question_mask):
        question_embeddings = self.embed(question)
        encoded = self.encoder(question_embeddings, question_mask)
        
        return encoded
    
    def decode(self, answer, answer_mask, encoded_question, question_mask):
        answer_embeddings = self.embed(answer)
        decoded = self.decoder(answer_embeddings, answer_mask, encoded_question, question_mask)
        
        return decoded
        
    def forward(self, question, question_mask, answer, answer_mask):
        encoded = self.encode(question, question_mask)
        decoded = self.decode(answer, answer_mask, encoded, question_mask)
    
        out = self.sequence_generator(decoded)
        out = F.log_softmax(out, dim=2)

        return out
