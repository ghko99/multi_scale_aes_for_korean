import os
import torch
from transformers import BertConfig, CONFIG_NAME, AutoTokenizer
from kobert_transformers import get_tokenizer
from document_bert_architectures import DocumentBertCombineWordDocumentLinear, DocumentBertSentenceChunkAttentionLSTM
from evaluate import quadratic_weighted_kappa_multi
from encoder import encode_documents_by_sentence, encode_documents_full_text
from torch.nn import functional as F
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import time
import warnings

# 개선된 손실 함수들
class AdaptiveLossWeighting(torch.nn.Module):
    """동적 손실 가중치 조정"""
    def __init__(self, num_tasks=3, init_weights=None):
        super().__init__()
        if init_weights is None:
            init_weights = [1.0, 0.5, 0.3]  # MSE, Sim, MR 초기 가중치
        self.log_weights = torch.nn.Parameter(torch.log(torch.tensor(init_weights)))
    
    def forward(self, losses):
        weights = torch.exp(self.log_weights)
        return torch.sum(weights * losses), weights

def improved_sim_loss(y, yhat, temperature=0.1):
    """개선된 코사인 유사도 손실 (온도 매개변수 추가)"""
    cos = F.cosine_similarity(y, yhat, dim=1)
    # 온도로 스케일링하여 더 민감하게 만들기
    loss = torch.mean((1 - cos) / temperature)
    return loss

def focal_mse_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal MSE Loss - 어려운 샘플에 더 집중"""
    mse = F.mse_loss(pred, target, reduction='none')
    # 평균 MSE로 정규화
    normalized_mse = mse / (torch.mean(mse) + 1e-8)
    # Focal weight 계산
    focal_weight = alpha * torch.pow(normalized_mse, gamma)
    return torch.mean(focal_weight * mse)

def improved_mr_loss_func(pred, label, margin=0.1):
    """개선된 순위 손실 (마진 추가)"""
    if label.size(0) <= 1:
        return torch.tensor(0.0, device=pred.device)
    
    total_mr_loss = 0
    num_criteria = pred.size(1)
    
    for criterion in range(num_criteria):
        pred_criterion = pred[:, criterion]
        label_criterion = label[:, criterion]
        
        mr_loss = 0
        pairs = 0
        
        for i in range(label.size(0)):
            for j in range(i + 1, label.size(0)):
                # 실제 순위 차이
                true_diff = label_criterion[i] - label_criterion[j]
                pred_diff = pred_criterion[i] - pred_criterion[j]
                
                # 순위가 뒤바뀐 경우에만 패널티
                if true_diff * pred_diff < 0:  # 부호가 다름
                    mr_loss += torch.clamp(margin - pred_diff * torch.sign(true_diff), min=0)
                    pairs += 1
        
        if pairs > 0:
            total_mr_loss += mr_loss / pairs
    
    return total_mr_loss / num_criteria if num_criteria > 0 else torch.tensor(0.0, device=pred.device)

class DocumentBertScoringModel():
    def __init__(self, load_model=False, chunk_model_path=None, word_doc_model_path=None, config=None, args=None):
        if args is not None:
            self.args = vars(args)
        else:
            # 개선된 기본 설정
            self.args = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'batch_size': 8,  # 배치 크기 증가
                'model_directory': './models',
                'result_file': 'result.txt',
                'gradient_accumulation_steps': 2,  # 그래디언트 누적
                'max_grad_norm': 1.0,
                'warmup_ratio': 0.1,
                'label_smoothing': 0.1
            }
            
        # 한국어 BERT 토크나이저 사용
        self.bert_tokenizer = get_tokenizer()
        
        # config 설정
        if config is None:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained("monologg/kobert")
        self.config = config
        
        # 문장별 처리를 위한 설정
        self.max_sentence_length = 128
        self.max_doc_length = 512
        
        # 11개 평가 기준 이름
        self.criterion_names = [
            '문법 정확도', '단어 선택의 적절성', '문장 표현', '문단 내 구조', '문단 간 구조',
            '구조의 일관성', '분량의 적절성', '주제 명료성', '창의성', 
            '프롬프트 독해력', '설명의 구체성'
        ]
        
        # 동적 손실 가중치
        self.adaptive_loss = AdaptiveLossWeighting().to(self.args['device'])
        
        print(f"Device: {self.args['device']}")
        print(f"Max sentence length: {self.max_sentence_length}")
        print(f"Max document length: {self.max_doc_length}")
        print(f"평가 기준: {self.criterion_names}")
        print(f"배치 크기: {self.args['batch_size']}")
        print(f"그래디언트 누적 스텝: {self.args['gradient_accumulation_steps']}")
        
        # 모델 로드 또는 초기화
        if load_model and chunk_model_path and word_doc_model_path:
            self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
                word_doc_model_path, config=config)
            self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
                chunk_model_path, config=config)
        else:
            self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear(bert_model_config=self.config)
            self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM(bert_model_config=self.config)

    def predict_for_regress(self, data, writeflag=False):
        """11개 평가 기준에 대한 멀티-태스크 회귀 예측"""
        correct_output = None
        
        if isinstance(data, tuple) and len(data) == 2:
            # 문장별 분할 인코딩
            document_representations_sentence, document_sequence_lengths_sentence = encode_documents_by_sentence(
                data[0], self.bert_tokenizer, max_input_length=self.max_sentence_length)
            
            # 전체 문서 인코딩
            document_representations_full, document_sequence_lengths_full = encode_documents_full_text(
                data[0], self.bert_tokenizer, max_input_length=self.max_doc_length)
            
            # 라벨 처리
            if isinstance(data[1], (list, np.ndarray)):
                if len(np.array(data[1]).shape) == 1:
                    labels_array = np.array(data[1])
                    if labels_array.shape[0] % 11 == 0:
                        correct_output = torch.FloatTensor(labels_array.reshape(-1, 11))
                    else:
                        raise ValueError("라벨 데이터 형태가 올바르지 않습니다. (N, 11) 형태여야 합니다.")
                else:
                    correct_output = torch.FloatTensor(data[1])
            else:
                correct_output = torch.FloatTensor(data[1])

        # 모델을 디바이스로 이동
        self.bert_regression_by_word_document.to(device=self.args['device'])
        self.bert_regression_by_chunk.to(device=self.args['device'])

        self.bert_regression_by_word_document.eval()
        self.bert_regression_by_chunk.eval()

        with torch.no_grad():
            eval_loss = 0
            count = 0
            predictions = torch.empty((document_representations_sentence.shape[0], 11))
            
            for i in range(0, document_representations_sentence.shape[0], self.args['batch_size']):
                # 배치 처리
                batch_document_tensors_sentence = document_representations_sentence[i:i + self.args['batch_size']].to(
                    device=self.args['device'])
                batch_predictions_sentence = self.bert_regression_by_chunk(
                    batch_document_tensors_sentence, device=self.args['device'])
                
                batch_document_tensors_full = document_representations_full[i:i + self.args['batch_size']].to(
                    device=self.args['device'])
                batch_predictions_full = self.bert_regression_by_word_document(
                    batch_document_tensors_full, device=self.args['device'])
                
                # 앙상블 예측 (가중 평균)
                batch_predictions_combined = 0.6 * batch_predictions_sentence + 0.4 * batch_predictions_full
                predictions[i:i + self.args['batch_size']] = batch_predictions_combined.cpu()
                
                # 손실 계산
                batch_labels = correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])
                
                # 개선된 손실 함수들 사용
                focal_mse = focal_mse_loss(batch_predictions_combined, batch_labels)
                sim_loss_val = improved_sim_loss(batch_predictions_combined, batch_labels)
                mr_loss_val = improved_mr_loss_func(batch_predictions_combined, batch_labels)
                
                losses = torch.stack([focal_mse, sim_loss_val, mr_loss_val])
                total_loss, weights = self.adaptive_loss(losses)
                
                eval_loss += total_loss.item()
                count += 1
                
            eval_loss /= count

        # 결과 저장
        if writeflag:
            outfile = open(os.path.join(self.args['model_directory'], self.args['result_file']), "w")
            for i in range(predictions.shape[0]):
                true_scores = correct_output[i].numpy()
                pred_scores = predictions[i].numpy()
                outfile.write(f"Sample {i}:\n")
                for j, criterion in enumerate(self.criterion_names):
                    outfile.write(f"  {criterion}: True={true_scores[j]:.3f}, Pred={pred_scores[j]:.3f}\n")
                outfile.write("\n")
            outfile.close()

        # 평가 메트릭 계산
        predictions_np = predictions.numpy()
        true_labels_np = correct_output.numpy()
        
        overall_mse = mean_squared_error(true_labels_np.flatten(), predictions_np.flatten())
        overall_mae = mean_absolute_error(true_labels_np.flatten(), predictions_np.flatten())
        overall_qwk = quadratic_weighted_kappa_multi(np.round(true_labels_np).astype(int).flatten(), 
                                                   np.round(predictions_np).astype(int).flatten())

        # 각 평가 기준별 메트릭
        criterion_mse = []
        criterion_mae = []
        criterion_qwk = []
        for i in range(11):
            mse = mean_squared_error(true_labels_np[:, i], predictions_np[:, i])
            mae = mean_absolute_error(true_labels_np[:, i], predictions_np[:, i])
            qwk = quadratic_weighted_kappa_multi(np.round(true_labels_np[:,i]).astype(int), 
                                               np.round(predictions_np[:,i]).astype(int))
            criterion_mse.append(mse)
            criterion_mae.append(mae)
            criterion_qwk.append(qwk)
        
        print(f"Overall MSE: {overall_mse:.4f} Overall MAE: {overall_mae:.4f} Overall QWK: {overall_qwk:.4f}")
        
        for i, criterion in enumerate(self.criterion_names):
            print(f"{criterion} - MSE: {criterion_mse[i]:.4f}, MAE: {criterion_mae[i]:.4f}, QWK: {criterion_qwk[i]:.4f}")
        
        return overall_mse, overall_mae, (true_labels_np, predictions_np), overall_qwk, eval_loss, criterion_mse, criterion_mae, criterion_qwk

    def fit(self, data_, test=None, mode='train', patience=8, log_dir='./logs'):
        """개선된 멀티-태스크 회귀 학습"""
        # 개선된 하이퍼파라미터
        lr = 2e-4  # 학습률 증가
        epochs = 20  # 에포크 수 증가
        weight_decay = 0.01
        warmup_steps_ratio = self.args['warmup_ratio']
        
        # 로그 설정
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'training_log_{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"개선된 Training Log - {mode} mode\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Max Epochs: {epochs}\n")
            f.write(f"Weight Decay: {weight_decay}\n")
            f.write(f"Patience: {patience}\n")
            f.write(f"Batch Size: {self.args['batch_size']}\n")
            f.write(f"Gradient Accumulation Steps: {self.args['gradient_accumulation_steps']}\n")
            f.write("="*80 + "\n\n")
        
        model_save_dir = './models'
        doc_model_save_dir = '{}/doc_model'.format(model_save_dir)
        chunk_model_save_dir = "{}/chunk_model".format(model_save_dir)
        
        # 개선된 옵티마이저 설정
        # 다른 학습률로 레이어별 설정
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.bert_regression_by_word_document.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr
            },
            {
                "params": [p for n, p in self.bert_regression_by_word_document.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr
            },
            {
                "params": [p for n, p in self.bert_regression_by_chunk.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr
            },
            {
                "params": [p for n, p in self.bert_regression_by_chunk.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr
            },
            {
                "params": self.adaptive_loss.parameters(),
                "weight_decay": 0.0,
                "lr": lr * 10  # 손실 가중치는 더 빠르게 학습
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        
        # 교차 검증
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        loss_list = []
        mse_list = []
        mae_list = []
        
        def log_message(message):
            print(message)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        
        for fold, (train_index, test_index) in enumerate(kf.split(data_[0])):
            log_message(f"=== Fold {fold + 1}/5 시작 ===")
            fold_start_time = time.time()
            
            train_essays = data_[0].iloc[train_index]
            train_labels = data_[1].iloc[train_index]
            
            test_essays = data_[0].iloc[test_index]
            test_labels = data_[1].iloc[test_index]
            
            data = train_essays.tolist(), train_labels.values
            test_data = test_essays.tolist(), test_labels.values
            
            log_message(f"훈련 데이터: {len(train_essays)}개, 검증 데이터: {len(test_essays)}개")
            
            # 데이터 인코딩
            log_message("데이터 인코딩 시작...")
            encoding_start_time = time.time()
            
            document_representations_sentence, _ = encode_documents_by_sentence(
                data[0], self.bert_tokenizer, max_input_length=self.max_sentence_length)
            document_representations_full, _ = encode_documents_full_text(
                data[0], self.bert_tokenizer, max_input_length=self.max_doc_length)
            
            correct_output = torch.FloatTensor(data[1])
            
            encoding_time = time.time() - encoding_start_time
            log_message(f"데이터 인코딩 완료 (소요시간: {encoding_time:.2f}초)")
            
            # 모델을 디바이스로 이동
            self.bert_regression_by_word_document.to(device=self.args['device'])
            self.bert_regression_by_chunk.to(device=self.args['device'])
            self.adaptive_loss.to(device=self.args['device'])
            
            self.bert_regression_by_word_document.train()
            self.bert_regression_by_chunk.train()
            self.adaptive_loss.train()
            
            # 학습률 스케줄러 설정
            total_steps = (len(data[0]) // (self.args['batch_size'] * self.args['gradient_accumulation_steps'])) * epochs
            warmup_steps = int(total_steps * warmup_steps_ratio)
            
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
            # Fold별 early stopping 초기화
            fold_best_qwk = -np.inf
            fold_best_eval_loss = np.inf
            fold_patience_counter = 0
            fold_early_stop = False
            
            # 학습 루프
            for epoch in range(1, epochs + 1):
                if fold_early_stop:
                    log_message(f"Early stopping triggered at epoch {epoch-1}")
                    break
                    
                epoch_start_time = time.time()
                epoch_loss = 0
                num_batches = 0
                
                log_message(f"--- Epoch {epoch}/{epochs} 시작 ---")
                
                # 그래디언트 누적을 위한 초기화
                optimizer.zero_grad()
                accumulated_loss = 0
                
                for i in range(0, document_representations_sentence.shape[0], self.args['batch_size']):
                    batch_start_time = time.time()
                    
                    # 배치 데이터 준비
                    batch_sentence = document_representations_sentence[i:i + self.args['batch_size']].to(
                        device=self.args['device'])
                    batch_full = document_representations_full[i:i + self.args['batch_size']].to(
                        device=self.args['device'])
                    batch_labels = correct_output[i:i + self.args['batch_size']].to(
                        device=self.args['device'])
                    
                    # Forward pass
                    predictions_sentence = self.bert_regression_by_chunk(batch_sentence, device=self.args['device'])
                    predictions_full = self.bert_regression_by_word_document(batch_full, device=self.args['device'])
                    
                    # 앙상블 예측
                    combined_predictions = 0.6 * predictions_sentence + 0.4 * predictions_full
                    
                    # 개선된 손실 계산
                    focal_mse = focal_mse_loss(combined_predictions, batch_labels)
                    sim_loss_val = improved_sim_loss(combined_predictions, batch_labels)
                    mr_loss_val = improved_mr_loss_func(combined_predictions, batch_labels)
                    
                    losses = torch.stack([focal_mse, sim_loss_val, mr_loss_val])
                    total_loss, current_weights = self.adaptive_loss(losses)
                    
                    # 그래디언트 누적을 위해 정규화
                    total_loss = total_loss / self.args['gradient_accumulation_steps']
                    total_loss.backward()
                    
                    accumulated_loss += total_loss.item()
                    
                    # 그래디언트 누적 스텝마다 업데이트
                    if (num_batches + 1) % self.args['gradient_accumulation_steps'] == 0:
                        # 그래디언트 클리핑
                        torch.nn.utils.clip_grad_norm_(
                            list(self.bert_regression_by_word_document.parameters()) +
                            list(self.bert_regression_by_chunk.parameters()) +
                            list(self.adaptive_loss.parameters()),
                            max_norm=self.args['max_grad_norm']
                        )
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        epoch_loss += accumulated_loss
                        accumulated_loss = 0
                    
                    num_batches += 1
                    batch_time = time.time() - batch_start_time
                    
                    # 배치별 상세 로깅 (매 20 배치마다)
                    if num_batches % 20 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        log_message(f"  Batch {num_batches}: Loss={total_loss.item()*self.args['gradient_accumulation_steps']:.4f} "
                                f"(Focal_MSE={focal_mse.item():.4f}, Sim={sim_loss_val.item():.4f}, "
                                f"MR={mr_loss_val.item():.4f}) "
                                f"Weights=[{current_weights[0]:.3f}, {current_weights[1]:.3f}, {current_weights[2]:.3f}] "
                                f"LR={current_lr:.2e} Time={batch_time:.2f}s")
                
                # 마지막 배치 처리
                if accumulated_loss > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.bert_regression_by_word_document.parameters()) +
                        list(self.bert_regression_by_chunk.parameters()) +
                        list(self.adaptive_loss.parameters()),
                        max_norm=self.args['max_grad_norm']
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    epoch_loss += accumulated_loss
                
                effective_batches = max(1, num_batches // self.args['gradient_accumulation_steps'])
                epoch_loss /= effective_batches
                loss_list.append(epoch_loss)
                epoch_time = time.time() - epoch_start_time
                
                current_lr = scheduler.get_last_lr()[0]
                log_message(f'Fold {fold + 1}, Epoch {epoch} 완료 - '
                        f'Loss: {epoch_loss:.4f}, '
                        f'LR: {current_lr:.2e}, '
                        f'Time: {epoch_time:.2f}s')
                
                # 검증
                if test:
                    eval_start_time = time.time()
                    log_message("검증 시작...")
                    
                    overall_mse, overall_mae, _, overall_qwk, eval_loss, criterion_mse, criterion_mae, criterion_qwk = self.predict_for_regress(test_data)
                    mse_list.append(overall_mse)
                    mae_list.append(overall_mae)
                    
                    eval_time = time.time() - eval_start_time
                    
                    log_message(f"검증 완료 (소요시간: {eval_time:.2f}초)")
                    log_message(f"Overall - MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}, "
                            f"QWK: {overall_qwk:.4f}, Eval Loss: {eval_loss:.4f}")
                    
                    # 각 평가 기준별 결과 로깅 (간소화)
                    avg_criterion_qwk = np.mean(criterion_qwk)
                    log_message(f"평균 기준별 QWK: {avg_criterion_qwk:.4f}")
                    
                    # 개선된 모델 저장 로직
                    save_flag = False
                    improvement_msg = ""
                    
                    # QWK 기준으로만 판단 (단순화)
                    if overall_qwk > fold_best_qwk:
                        fold_best_qwk = overall_qwk
                        fold_best_eval_loss = eval_loss
                        save_flag = True
                        improvement_msg = f"QWK 개선 ({overall_qwk:.4f})"
                        fold_patience_counter = 0
                    else:
                        fold_patience_counter += 1
                        improvement_msg = f"성능 개선 없음 (patience: {fold_patience_counter}/{patience})"
                    
                    if save_flag:
                        if not os.path.exists(model_save_dir):
                            os.makedirs(model_save_dir)
                        self.bert_regression_by_word_document.save_pretrained(doc_model_save_dir)
                        self.bert_regression_by_chunk.save_pretrained(chunk_model_save_dir)
                        torch.save(self.adaptive_loss.state_dict(), os.path.join(model_save_dir, 'adaptive_loss.pt'))
                        log_message(f"모델 저장 완료: {improvement_msg}")
                    else:
                        log_message(improvement_msg)
                    
                    # Early stopping 체크
                    if fold_patience_counter >= patience:
                        log_message(f"Early stopping: {patience} epochs 동안 성능 개선 없음")
                        fold_early_stop = True
                    
                    # 다시 학습 모드로 변경
                    self.bert_regression_by_word_document.train()
                    self.bert_regression_by_chunk.train()
                    self.adaptive_loss.train()
            
            fold_time = time.time() - fold_start_time
            log_message(f"=== Fold {fold + 1} 완료 (총 소요시간: {fold_time:.2f}초) ===\n")
        
        # 전체 학습 완료 후 결과 저장
        os.makedirs('./train_valid_loss', exist_ok=True)
        np.save(f'./train_valid_loss/{mode}_loss.npy', np.array(loss_list))
        np.save(f'./train_valid_loss/{mode}_mse.npy', np.array(mse_list))
        np.save(f'./train_valid_loss/{mode}_mae.npy', np.array(mae_list))
        
        final_mse = np.mean(mse_list) if mse_list else float('inf')
        final_mae = np.mean(mae_list) if mae_list else float('inf')
        
        log_message("="*80)
        log_message("개선된 학습 완료!")
        log_message(f"최종 평균 MSE: {final_mse:.4f}")
        log_message(f"최종 평균 MAE: {final_mae:.4f}")
        log_message(f"로그 파일 저장 위치: {log_file}")
        log_message(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"Average MSE: {final_mse:.4f}")
        print(f"Average MAE: {final_mae:.4f}")
        print(f"Training log saved to: {log_file}")

    def predict_single(self, input_sentence):
        """단일 문장에 대한 11개 평가 기준 예측"""
        # 문장별 인코딩
        document_representations_sentence, _ = encode_documents_by_sentence(
            [input_sentence], self.bert_tokenizer, max_input_length=self.max_sentence_length)
        
        # 전체 문서 인코딩
        document_representations_full, _ = encode_documents_full_text(
            [input_sentence], self.bert_tokenizer, max_input_length=self.max_doc_length)
        
        # 모델을 디바이스로 이동
        self.bert_regression_by_word_document.to(device=self.args['device'])
        self.bert_regression_by_chunk.to(device=self.args['device'])
        
        self.bert_regression_by_word_document.eval()
        self.bert_regression_by_chunk.eval()
        
        with torch.no_grad():
            # 예측
            sentence_tensor = document_representations_sentence.to(device=self.args['device'])
            full_tensor = document_representations_full.to(device=self.args['device'])
            
            predictions_sentence = self.bert_regression_by_chunk(sentence_tensor, device=self.args['device'])
            predictions_full = self.bert_regression_by_word_document(full_tensor, device=self.args['device'])
            
            # 앙상블 결합 (개선된 가중치)
            combined_predictions = 0.6 * predictions_sentence + 0.4 * predictions_full
            predictions = combined_predictions.cpu().numpy()[0]  # (11,)
        
        # 결과를 딕셔너리로 반환
        result_dict = {}
        for i, criterion in enumerate(self.criterion_names):
            result_dict[criterion] = float(predictions[i])
        
        return result_dict, predictions
    
    def load_best_model(self, model_dir='./models'):
        """저장된 최고 성능 모델 로드"""
        doc_model_path = os.path.join(model_dir, 'doc_model')
        chunk_model_path = os.path.join(model_dir, 'chunk_model')
        adaptive_loss_path = os.path.join(model_dir, 'adaptive_loss.pt')
        
        if os.path.exists(doc_model_path) and os.path.exists(chunk_model_path):
            print("저장된 모델을 로드하는 중...")
            self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
                doc_model_path, config=self.config)
            self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
                chunk_model_path, config=self.config)
            
            if os.path.exists(adaptive_loss_path):
                self.adaptive_loss.load_state_dict(torch.load(adaptive_loss_path))
                print("적응적 손실 가중치도 로드되었습니다.")
            
            print("모델 로드 완료!")
        else:
            print("저장된 모델을 찾을 수 없습니다. 기본 모델을 사용합니다.")
    
    def analyze_training_progress(self, mode='train'):
        """학습 진행 상황 분석 및 시각화"""
        try:
            import matplotlib.pyplot as plt
            
            loss_path = f'./train_valid_loss/{mode}_loss.npy'
            mse_path = f'./train_valid_loss/{mode}_mse.npy'
            mae_path = f'./train_valid_loss/{mode}_mae.npy'
            
            if all(os.path.exists(p) for p in [loss_path, mse_path, mae_path]):
                losses = np.load(loss_path)
                mses = np.load(mse_path)
                maes = np.load(mae_path)
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].plot(losses)
                axes[0].set_title('Training Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].grid(True)
                
                axes[1].plot(mses)
                axes[1].set_title('Validation MSE')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('MSE')
                axes[1].grid(True)
                
                axes[2].plot(maes)
                axes[2].set_title('Validation MAE')
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('MAE')
                axes[2].grid(True)
                
                plt.tight_layout()
                plt.savefig(f'./logs/training_progress_{mode}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"학습 진행 상황 그래프가 저장되었습니다: ./logs/training_progress_{mode}.png")
            else:
                print("학습 기록 파일을 찾을 수 없습니다.")
                
        except ImportError:
            print("matplotlib이 설치되지 않아 시각화를 건너뜁니다.")
        except Exception as e:
            print(f"시각화 중 오류 발생: {e}")