import torch
import math
import logging
from transformers import BertTokenizer, AutoTokenizer
import gc

log = logging.getLogger()


def encode_documents_by_sentence_batch(documents: list, tokenizer, max_input_length=128, 
                                     sentence_delimiter="#@문장구분#", batch_size=32):
    """
    메모리 효율적인 문장별 분할 인코딩 함수 (배치 처리)
    
    Args:
        documents: 문서 리스트
        tokenizer: BERT 토크나이저
        max_input_length: 각 문장의 최대 토큰 길이
        sentence_delimiter: 문장 구분자
        batch_size: 배치 크기 (메모리 사용량 조절)
    
    Returns:
        output: (num_documents, max_sentences, 3, max_input_length) 형태의 텐서
        document_seq_lengths: 각 문서의 실제 문장 수
    """
    total_docs = len(documents)
    log.info(f"문장별 인코딩 시작: 총 {total_docs}개 문서, 배치 크기: {batch_size}")
    
    # 첫 번째 패스: 최대 문장 수 계산 (메모리 절약)
    max_sentences = 0
    doc_sentence_counts = []
    
    for i, document in enumerate(documents):
        if i % 1000 == 0:
            log.info(f"문장 수 계산 진행: {i}/{total_docs}")
        
        sentences = document.split(sentence_delimiter)
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        sentence_count = len(sentences)
        
        doc_sentence_counts.append(sentence_count)
        max_sentences = max(max_sentences, sentence_count)
    
    log.info(f"최대 문장 수: {max_sentences}")
    
    # 출력 텐서 초기화
    output = torch.zeros(size=(total_docs, max_sentences, 3, max_input_length), dtype=torch.long)
    document_seq_lengths = torch.LongTensor(doc_sentence_counts)
    
    # 배치별 처리
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        log.info(f"배치 처리 중: {batch_start}-{batch_end}/{total_docs}")
        
        # 현재 배치 처리
        for doc_idx in range(batch_start, batch_end):
            document = documents[doc_idx]
            sentences = document.split(sentence_delimiter)
            sentences = [sent.strip() for sent in sentences if sent.strip()]
            
            for sent_idx, sentence in enumerate(sentences):
                # 토큰화 및 길이 제한
                tokens = tokenizer.tokenize(sentence)
                if len(tokens) > max_input_length - 2:
                    tokens = tokens[:max_input_length - 2]
                
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
                output[doc_idx][sent_idx] = torch.cat((
                    torch.LongTensor(input_ids).unsqueeze(0),
                    torch.LongTensor(input_type_ids).unsqueeze(0),
                    torch.LongTensor(attention_masks).unsqueeze(0)
                ), dim=0)
        
        # 가비지 컬렉션으로 메모리 정리
        gc.collect()
    
    return output, document_seq_lengths


def encode_documents_full_text_batch(documents: list, tokenizer, max_input_length=512, batch_size=32):
    """
    메모리 효율적인 전체 문서 인코딩 함수 (배치 처리)
    
    Args:
        documents: 문서 리스트
        tokenizer: BERT 토크나이저
        max_input_length: 최대 토큰 길이
        batch_size: 배치 크기
    
    Returns:
        output: (num_documents, 1, 3, max_input_length) 형태의 텐서
        document_seq_lengths: 각 문서의 실제 토큰 수
    """
    total_docs = len(documents)
    log.info(f"전체 문서 인코딩 시작: 총 {total_docs}개 문서, 배치 크기: {batch_size}")
    
    # 출력 텐서 초기화
    output = torch.zeros(size=(total_docs, 1, 3, max_input_length), dtype=torch.long)
    document_seq_lengths = torch.ones(total_docs, dtype=torch.long)
    
    # 배치별 처리
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        log.info(f"배치 처리 중: {batch_start}-{batch_end}/{total_docs}")
        
        for doc_idx in range(batch_start, batch_end):
            document = documents[doc_idx]
            
            # 문장 구분자 제거하고 전체 텍스트로 처리
            clean_document = document.replace("#@문장구분#", " ")
            tokens = tokenizer.tokenize(clean_document)
            
            # 문서가 너무 길면 잘라내기
            if len(tokens) > max_input_length - 2:
                tokens = tokens[:max_input_length - 2]
            
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
            output[doc_idx][0] = torch.cat((
                torch.LongTensor(input_ids).unsqueeze(0),
                torch.LongTensor(input_type_ids).unsqueeze(0),
                torch.LongTensor(attention_masks).unsqueeze(0)
            ), dim=0)
        
        # 가비지 컬렉션으로 메모리 정리
        gc.collect()
    
    return output, document_seq_lengths


def encode_documents_by_sentence_streaming(documents: list, tokenizer, max_input_length=128, 
                                         sentence_delimiter="#@문장구분#"):
    """
    스트리밍 방식의 문장별 분할 인코딩 (초대용량 데이터용)
    
    Generator 패턴을 사용하여 메모리 사용량을 최소화
    """
    for doc_idx, document in enumerate(documents):
        sentences = document.split(sentence_delimiter)
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        
        doc_tensors = []
        for sent_idx, sentence in enumerate(sentences):
            tokens = tokenizer.tokenize(sentence)
            if len(tokens) > max_input_length - 2:
                tokens = tokens[:max_input_length - 2]
            
            full_tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_type_ids = [0] * len(full_tokens)
            
            input_ids = tokenizer.convert_tokens_to_ids(full_tokens)
            attention_masks = [1] * len(input_ids)
            
            # 패딩 추가
            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)
            
            sentence_tensor = torch.cat((
                torch.LongTensor(input_ids).unsqueeze(0),
                torch.LongTensor(input_type_ids).unsqueeze(0),
                torch.LongTensor(attention_masks).unsqueeze(0)
            ), dim=0)
            
            doc_tensors.append(sentence_tensor)
        
        yield doc_idx, doc_tensors, len(sentences)


# 기존 함수들 (메모리 최적화)
def encode_documents_by_sentence(documents: list, tokenizer, max_input_length=128, 
                               sentence_delimiter="#@문장구분#"):
    """
    기존 호환성을 위한 래퍼 함수 - 배치 처리 버전 사용
    """
    # 문서 수에 따라 배치 크기 동적 조정
    num_docs = len(documents)
    if num_docs > 1000:
        batch_size = 16  # 대용량 데이터의 경우 작은 배치
    elif num_docs > 500:
        batch_size = 32
    else:
        batch_size = 64
    
    return encode_documents_by_sentence_batch(
        documents, tokenizer, max_input_length, sentence_delimiter, batch_size)


def encode_documents_full_text(documents: list, tokenizer, max_input_length=512):
    """
    기존 호환성을 위한 래퍼 함수 - 배치 처리 버전 사용
    """
    # 문서 수에 따라 배치 크기 동적 조정
    num_docs = len(documents)
    if num_docs > 1000:
        batch_size = 8   # 전체 문서는 더 큰 메모리를 사용하므로 작은 배치
    elif num_docs > 500:
        batch_size = 16
    else:
        batch_size = 32
    
    return encode_documents_full_text_batch(documents, tokenizer, max_input_length, batch_size)


def encode_documents(documents: list, tokenizer, max_input_length=512, use_sentence_split=False):
    """
    문서 인코딩 통합 함수 (메모리 최적화 버전)
    
    Args:
        use_sentence_split: True면 문장별 분할, False면 전체 텍스트 처리
    """
    # 메모리 정리
    gc.collect()
    
    if use_sentence_split:
        return encode_documents_by_sentence(documents, tokenizer, max_input_length)
    else:
        return encode_documents_full_text(documents, tokenizer, max_input_length)


def get_memory_usage():
    """현재 메모리 사용량 반환 (디버깅용)"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return f"{memory_mb:.2f} MB"


def log_memory_usage(message=""):
    """메모리 사용량 로깅"""
    try:
        memory_usage = get_memory_usage()
        log.info(f"{message} - 메모리 사용량: {memory_usage}")
    except ImportError:
        log.info(f"{message} - psutil 없음: 메모리 사용량 측정 불가")