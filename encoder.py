import torch
import math
import logging
from transformers import BertTokenizer, AutoTokenizer

log = logging.getLogger()


def encode_documents_by_sentence(documents: list, tokenizer, max_input_length=128, sentence_delimiter="#@문장구분#"):
    """
    한국어 에세이를 문장별로 분할하여 인코딩하는 함수
    
    Args:
        documents: 문서 리스트
        tokenizer: BERT 토크나이저
        max_input_length: 각 문장의 최대 토큰 길이 (기본값: 128)
        sentence_delimiter: 문장 구분자 (기본값: "#@문장구분#")
    
    Returns:
        output: (num_documents, max_sentences, 3, max_input_length) 형태의 텐서
        document_seq_lengths: 각 문서의 실제 문장 수
    """
    tokenized_documents = []
    max_sentences = 0
    
    # 각 문서를 문장별로 분할하고 토크나이징
    for document in documents:
        sentences = document.split(sentence_delimiter)
        # 빈 문장 제거
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        
        tokenized_sentences = []
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            # 문장이 너무 길면 잘라내기
            if len(tokens) > max_input_length - 2:  # [CLS], [SEP] 토큰 고려
                tokens = tokens[:max_input_length - 2]
            tokenized_sentences.append(tokens)
        
        tokenized_documents.append(tokenized_sentences)
        max_sentences = max(max_sentences, len(tokenized_sentences))
    
    # 출력 텐서 초기화
    output = torch.zeros(size=(len(documents), max_sentences, 3, max_input_length), dtype=torch.long)
    document_seq_lengths = []
    
    # 각 문서 처리
    for doc_index, tokenized_sentences in enumerate(tokenized_documents):
        for sent_index, tokens in enumerate(tokenized_sentences):
            # [CLS] 토큰 추가
            full_tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_type_ids = [0] * len(full_tokens)
            
            # 토큰을 ID로 변환
            input_ids = tokenizer.convert_tokens_to_ids(full_tokens)
            attention_masks = [1] * len(input_ids)
            
            # 패딩 추가
            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)
            
            # 텐서에 저장
            output[doc_index][sent_index] = torch.cat((
                torch.LongTensor(input_ids).unsqueeze(0),
                torch.LongTensor(input_type_ids).unsqueeze(0),
                torch.LongTensor(attention_masks).unsqueeze(0)
            ), dim=0)
        
        document_seq_lengths.append(len(tokenized_sentences))
    
    return output, torch.LongTensor(document_seq_lengths)


def encode_documents_full_text(documents: list, tokenizer, max_input_length=512):
    """
    전체 문서를 하나의 시퀀스로 인코딩하는 함수 (DocumentBertCombineWordDocumentLinear용)
    
    Args:
        documents: 문서 리스트
        tokenizer: BERT 토크나이저
        max_input_length: 최대 토큰 길이
    
    Returns:
        output: (num_documents, 1, 3, max_input_length) 형태의 텐서
        document_seq_lengths: 각 문서의 실제 토큰 수
    """
    tokenized_documents = []
    
    for document in documents:
        # 문장 구분자 제거하고 전체 텍스트로 처리
        clean_document = document.replace("#@문장구분#", " ")
        tokens = tokenizer.tokenize(clean_document)
        
        # 문서가 너무 길면 잘라내기
        if len(tokens) > max_input_length - 2:
            tokens = tokens[:max_input_length - 2]
        
        tokenized_documents.append(tokens)
    
    # 출력 텐서 초기화
    output = torch.zeros(size=(len(documents), 1, 3, max_input_length), dtype=torch.long)
    document_seq_lengths = []
    
    # 각 문서 처리
    for doc_index, tokens in enumerate(tokenized_documents):
        # [CLS], [SEP] 토큰 추가
        full_tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_type_ids = [0] * len(full_tokens)
        
        # 토큰을 ID로 변환
        input_ids = tokenizer.convert_tokens_to_ids(full_tokens)
        attention_masks = [1] * len(input_ids)
        
        # 패딩 추가
        while len(input_ids) < max_input_length:
            input_ids.append(0)
            input_type_ids.append(0)
            attention_masks.append(0)
        
        # 텐서에 저장
        output[doc_index][0] = torch.cat((
            torch.LongTensor(input_ids).unsqueeze(0),
            torch.LongTensor(input_type_ids).unsqueeze(0),
            torch.LongTensor(attention_masks).unsqueeze(0)
        ), dim=0)
        
        document_seq_lengths.append(1)  # 항상 1개의 시퀀스
    
    return output, torch.LongTensor(document_seq_lengths)


# 기존 함수와의 호환성을 위한 래퍼 함수
def encode_documents(documents: list, tokenizer, max_input_length=512, use_sentence_split=False):
    """
    문서 인코딩 통합 함수
    
    Args:
        use_sentence_split: True면 문장별 분할, False면 전체 텍스트 처리
    """
    if use_sentence_split:
        return encode_documents_by_sentence(documents, tokenizer, max_input_length)
    else:
        return encode_documents_full_text(documents, tokenizer, max_input_length)