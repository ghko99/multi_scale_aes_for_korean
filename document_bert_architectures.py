import torch
from torch import nn
from torch.nn import LSTM
from transformers import BertPreTrainedModel, BertConfig, BertModel
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)  # 0~3 범위에 맞게 작은 값으로 초기화


class DocumentBertSentenceChunkAttentionLSTM(BertPreTrainedModel):  
    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertSentenceChunkAttentionLSTM, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)  
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.lstm = LSTM(bert_model_config.hidden_size, bert_model_config.hidden_size, batch_first=True)
        
        # 11개의 평가 기준을 위한 멀티-태스크 회귀
        self.mlp = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size // 2, 11)  # 11개 평가 기준
        )
        
        # Attention mechanism parameters
        self.w_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, bert_model_config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, bert_model_config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, device='cpu'):
        """
        document_batch: (batch_size, max_sentences, 3, max_seq_length)
        Returns: (batch_size, 11) - 11개 평가 기준의 점수 (0~3)
        """
        batch_size, max_sentences = document_batch.shape[0], document_batch.shape[1]
        
        # BERT 임베딩 계산
        bert_output = torch.zeros(size=(batch_size, max_sentences, self.bert.config.hidden_size), 
                                 dtype=torch.float, device=device)
        
        for doc_id in range(batch_size):
            for sent_id in range(max_sentences):
                # 패딩된 문장은 건너뛰기 (모든 토큰이 0인 경우)
                if torch.sum(document_batch[doc_id][sent_id][0]) == 0:
                    continue
                    
                # BERT forward pass
                bert_outputs = self.bert(
                    input_ids=document_batch[doc_id][sent_id][0].unsqueeze(0),
                    token_type_ids=document_batch[doc_id][sent_id][1].unsqueeze(0),
                    attention_mask=document_batch[doc_id][sent_id][2].unsqueeze(0)
                )
                # [CLS] 토큰의 임베딩 사용
                bert_output[doc_id][sent_id] = self.dropout(bert_outputs.pooler_output.squeeze(0))
        
        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(bert_output)
        
        # Attention mechanism
        attention_w = torch.tanh(torch.matmul(lstm_output, self.w_omega) + self.b_omega)
        attention_u = torch.matmul(attention_w, self.u_omega)  # (batch_size, max_sentences, 1)
        attention_score = F.softmax(attention_u, dim=1)  # (batch_size, max_sentences, 1)
        attention_hidden = lstm_output * attention_score  # (batch_size, max_sentences, hidden_size)
        attention_hidden = torch.sum(attention_hidden, dim=1)  # (batch_size, hidden_size)
        
        # 11개 평가 기준 예측
        prediction = self.mlp(attention_hidden)
        
        # 0~3 범위로 클리핑
        prediction = torch.clamp(prediction, min=0.0, max=3.0)
        
        return prediction


class DocumentBertCombineWordDocumentLinear(BertPreTrainedModel): 
    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertCombineWordDocumentLinear, self).__init__(bert_model_config)
        
        self.bert = BertModel(bert_model_config)
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        # 11개의 평가 기준을 위한 멀티-태스크 회귀
        self.mlp = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size * 2, bert_model_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, 11)  # 11개 평가 기준
        )
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, device='cpu'):
        """
        document_batch: (batch_size, 1, 3, max_seq_length) - 전체 문서를 하나의 시퀀스로 처리
        Returns: (batch_size, 11) - 11개 평가 기준의 점수 (0~3)
        """
        batch_size = document_batch.shape[0]
        bert_output = torch.zeros(size=(batch_size, self.bert.config.hidden_size * 2),
                                 dtype=torch.float, device=device)
        
        for doc_id in range(batch_size):
            # BERT forward pass
            all_bert_output_info = self.bert(
                input_ids=document_batch[doc_id][0][0].unsqueeze(0),
                token_type_ids=document_batch[doc_id][0][1].unsqueeze(0),
                attention_mask=document_batch[doc_id][0][2].unsqueeze(0)
            )
            
            # Max pooling over token embeddings + [CLS] token
            bert_token_max = torch.max(all_bert_output_info.last_hidden_state, dim=1)
            bert_output[doc_id] = torch.cat((bert_token_max.values.squeeze(0), 
                                           all_bert_output_info.pooler_output.squeeze(0)), dim=0)

        prediction = self.mlp(bert_output)
        
        # 0~3 범위로 클리핑
        prediction = torch.clamp(prediction, min=0.0, max=3.0)
        
        return prediction