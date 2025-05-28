import os
import torch
from transformers import BertConfig, CONFIG_NAME, AutoTokenizer
from document_bert_architectures import DocumentBertCombineWordDocumentLinear, DocumentBertSentenceChunkAttentionLSTM
from evaluate import quadratic_weighted_kappa_multi
from encoder import encode_documents_by_sentence, encode_documents_full_text
from torch.nn import functional as F
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import time

# 코사인 유사도 기반 손실
def sim_loss(y, yhat):
    cos = F.cosine_similarity(y, yhat, dim=1)
    loss = torch.mean(1 - cos)
    return loss

# 순위 손실 (각 평가 기준별로 적용)
def mr_loss_func(pred, label):
    if label.size(0) <= 1:
        return torch.tensor(0.0, device=pred.device)
    
    total_mr_loss = 0
    num_criteria = pred.size(1)  # 11개 평가 기준
    
    for criterion in range(num_criteria):
        pred_criterion = pred[:, criterion]
        label_criterion = label[:, criterion]
        
        mr_loss = 0
        for i in range(label.size(0)):
            y = pred_criterion - pred_criterion[i]
            yhat = label_criterion - label_criterion[i]
            yhat = yhat.sign()
            mask = y.sign() != yhat.sign()
            mr_loss += y[mask].abs().sum()
        
        total_mr_loss += mr_loss / (label.size(0) * (label.size(0) - 1))
    
    return total_mr_loss / num_criteria

class DocumentBertScoringModel():
    def __init__(self, load_model=False, chunk_model_path=None, word_doc_model_path=None, config=None, args=None):
        if args is not None:
            self.args = vars(args)
        else:
            # 기본 설정
            self.args = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'batch_size': 4,
                'model_directory': './models',
                'result_file': 'result.txt'
            }
            
        # 한국어 BERT 토크나이저 사용
        self.bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        
        # config 설정 (KLUE-BERT 기본 설정 사용)
        if config is None:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained("klue/bert-base")
        self.config = config
        
        # 문장별 처리를 위한 설정
        self.max_sentence_length = 128  # 각 문장의 최대 토큰 수
        self.max_doc_length = 512  # 전체 문서의 최대 토큰 수
        
        # 11개 평가 기준 이름 (예시)
        self.criterion_names = [
            '문법 정확도', '단어 선택의 적절성', '문장 표현', '문단 내 구조', '문단 간 구조',
            '구조의 일관성', '분량의 적절성', '주제 명료성', '창의성', 
            '프롬프트 독해력', '설명의 구체성'
        ]
        
        print(f"Device: {self.args['device']}")
        print(f"Max sentence length: {self.max_sentence_length}")
        print(f"Max document length: {self.max_doc_length}")
        print(f"평가 기준: {self.criterion_names}")
        
        # 모델 로드 또는 초기화
        if load_model and chunk_model_path and word_doc_model_path:
            self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
                word_doc_model_path, config=config)
            self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
                chunk_model_path, config=config)
        else:
            # KLUE-BERT로 초기화
            self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
                "klue/bert-base")
            self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
                "klue/bert-base")

    def predict_for_regress(self, data, writeflag=False):
        """11개 평가 기준에 대한 멀티-태스크 회귀 예측"""
        correct_output = None
        
        if isinstance(data, tuple) and len(data) == 2:
            # 문장별 분할 인코딩 (LSTM 모델용)
            document_representations_sentence, document_sequence_lengths_sentence = encode_documents_by_sentence(
                data[0], self.bert_tokenizer, max_input_length=self.max_sentence_length)
            
            # 전체 문서 인코딩 (Linear 모델용)
            document_representations_full, document_sequence_lengths_full = encode_documents_full_text(
                data[0], self.bert_tokenizer, max_input_length=self.max_doc_length)
            
            # 라벨이 (N, 11) 형태인지 확인
            if isinstance(data[1], (list, np.ndarray)):
                if len(np.array(data[1]).shape) == 1:
                    # 1D 배열인 경우 각 샘플이 11개 값을 가진다고 가정하고 reshape
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
            predictions = torch.empty((document_representations_sentence.shape[0], 11))  # (N, 11)
            
            for i in range(0, document_representations_sentence.shape[0], self.args['batch_size']):
                # 문장별 처리 모델
                batch_document_tensors_sentence = document_representations_sentence[i:i + self.args['batch_size']].to(
                    device=self.args['device'])
                batch_predictions_sentence = self.bert_regression_by_chunk(
                    batch_document_tensors_sentence, device=self.args['device'])
                
                # 전체 문서 처리 모델
                batch_document_tensors_full = document_representations_full[i:i + self.args['batch_size']].to(
                    device=self.args['device'])
                batch_predictions_full = self.bert_regression_by_word_document(
                    batch_document_tensors_full, device=self.args['device'])
                
                # 두 모델의 출력을 평균
                batch_predictions_combined = (batch_predictions_sentence + batch_predictions_full) / 2
                predictions[i:i + self.args['batch_size']] = batch_predictions_combined.cpu()
                
                # 손실 계산
                batch_labels = correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])
                mse_loss = F.mse_loss(batch_predictions_combined, batch_labels)
                sim_loss_val = sim_loss(batch_predictions_combined, batch_labels)
                mr_loss_val = mr_loss_func(batch_predictions_combined, batch_labels)
                
                # 가중 손실
                a, b, c = 1, 1, 2  # 가중치
                total_loss = a * mse_loss + b * sim_loss_val + c * mr_loss_val
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
        
        # 전체 MSE, MAE
        overall_mse = mean_squared_error(true_labels_np.flatten(), predictions_np.flatten())
        overall_mae = mean_absolute_error(true_labels_np.flatten(), predictions_np.flatten())
        overall_qwk = quadratic_weighted_kappa_multi(np.round(true_labels_np).astype(int).flatten() , np.round(predictions_np).astype(int).flatten())


        # 각 평가 기준별 MSE, MAE
        criterion_mse = []
        criterion_mae = []
        criterion_qwk = []
        for i in range(11):
            mse = mean_squared_error(true_labels_np[:, i], predictions_np[:, i])
            mae = mean_absolute_error(true_labels_np[:, i], predictions_np[:, i])
            qwk = quadratic_weighted_kappa_multi(np.round(true_labels_np[:,i]).astype(int), np.round(predictions_np[:,i]).astype(int))
            criterion_mse.append(mse)
            criterion_mae.append(mae)
            criterion_qwk.append(qwk)
        
        print(f"Overall MSE: {overall_mse:.4f} Overall MAE: {overall_mae:.4f} Overall QWK: {overall_qwk:.4f}")
        
        for i, criterion in enumerate(self.criterion_names):
            print(f"{criterion} - MSE: {criterion_mse[i]:.4f}, MAE: {criterion_mae[i]:.4f}, QWK: {criterion_qwk[i]:.4f}")
        
        return overall_mse, overall_mae, (true_labels_np, predictions_np), overall_qwk, eval_loss, criterion_mse, criterion_mae, criterion_qwk

    def fit(self, data_, test=None, mode='train', patience=5, log_dir='./logs'):
        """11개 평가 기준에 대한 멀티-태스크 회귀 학습 (로깅 및 early stopping 추가)"""
        lr = 6e-5
        epochs = 16
        weight_decay = 0.005
        
        # 로그 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'training_log_{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        # 로그 파일 초기화
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Training Log - {mode} mode\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Max Epochs: {epochs}\n")
            f.write(f"Weight Decay: {weight_decay}\n")
            f.write(f"Patience: {patience}\n")
            f.write(f"Batch Size: {self.args['batch_size']}\n")
            f.write("="*80 + "\n\n")
        
        model_save_dir = './models'
        doc_model_save_dir = '{}/doc_model'.format(model_save_dir)
        chunk_model_save_dir = "{}/chunk_model".format(model_save_dir)
        
        # 옵티마이저
        word_document_optimizer = torch.optim.Adam(
            self.bert_regression_by_word_document.parameters(), lr=lr, weight_decay=weight_decay)
        chunk_optimizer = torch.optim.Adam(
            self.bert_regression_by_chunk.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 학습률 스케줄러
        word_document_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            word_document_optimizer, T_max=epochs)
        chunk_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            chunk_optimizer, T_max=epochs)
        
        # 교차 검증
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        loss_list = []
        mse_list = []
        mae_list = []
        
        def log_message(message):
            """로그 메시지를 파일과 콘솔에 동시 출력"""
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
            
            self.bert_regression_by_word_document.train()
            self.bert_regression_by_chunk.train()
            
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
                    
                    # 두 모델의 출력을 결합
                    combined_predictions = (predictions_sentence + predictions_full) / 2
                    
                    # 손실 계산
                    mse_loss = F.mse_loss(combined_predictions, batch_labels)
                    sim_loss_val = sim_loss(combined_predictions, batch_labels)
                    mr_loss_val = mr_loss_func(combined_predictions, batch_labels)
                    
                    # 가중 손실
                    a, b, c = 1, 1, 2
                    total_loss = a * mse_loss + b * sim_loss_val + c * mr_loss_val
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(self.bert_regression_by_word_document.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.bert_regression_by_chunk.parameters(), max_norm=1.0)
                    
                    # 옵티마이저 스텝
                    word_document_optimizer.step()
                    chunk_optimizer.step()
                    
                    word_document_optimizer.zero_grad()
                    chunk_optimizer.zero_grad()
                    
                    epoch_loss += total_loss.item()
                    num_batches += 1
                    
                    batch_time = time.time() - batch_start_time
                    
                    # 배치별 상세 로깅 (매 10 배치마다)
                    if num_batches % 10 == 0:
                        log_message(f"  Batch {num_batches}: Loss={total_loss.item():.4f} "
                                f"(MSE={mse_loss.item():.4f}, Sim={sim_loss_val.item():.4f}, "
                                f"MR={mr_loss_val.item():.4f}) Time={batch_time:.2f}s")
                
                epoch_loss /= num_batches
                loss_list.append(epoch_loss)
                epoch_time = time.time() - epoch_start_time
                
                # 학습률 업데이트
                word_document_scheduler.step()
                chunk_scheduler.step()
                
                current_lr_word = word_document_scheduler.get_last_lr()[0]
                current_lr_chunk = chunk_scheduler.get_last_lr()[0]
                
                log_message(f'Fold {fold + 1}, Epoch {epoch} 완료 - '
                        f'Loss: {epoch_loss:.4f}, '
                        f'LR_word: {current_lr_word:.2e}, '
                        f'LR_chunk: {current_lr_chunk:.2e}, '
                        f'Time: {epoch_time:.2f}s')
                
                # 검증
                if test:
                    eval_start_time = time.time()
                    log_message("검증 시작...")
                    
                    overall_mse, overall_mae, _, overall_qwk, eval_loss, criterion_mse, criterion_mae, criterion_qwk = self.predict_for_regress(test_data)
                    mse_list.append(overall_mse)
                    mae_list.append(overall_mae)
                    
                    eval_time = time.time() - eval_start_time
                    
                    # 평가 기준별 상세 로깅
                    log_message(f"검증 완료 (소요시간: {eval_time:.2f}초)")
                    log_message(f"Overall - MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}, "
                            f"QWK: {overall_qwk:.4f}, Eval Loss: {eval_loss:.4f}")
                    
                    # 각 평가 기준별 결과 로깅
                    for i, criterion in enumerate(self.criterion_names):
                        log_message(f"  {criterion}: MSE={criterion_mse[i]:.4f}, "
                                f"MAE={criterion_mae[i]:.4f}, QWK={criterion_qwk[i]:.4f}")
                    
                    new_qwk = np.mean([overall_qwk] + criterion_qwk)
                    save_flag = False
                    improvement_msg = ""
                    
                    # 모델 저장 조건 체크
                    if eval_loss < fold_best_eval_loss:
                        fold_best_eval_loss = eval_loss
                        save_flag = True
                        improvement_msg += f"Eval Loss 개선 ({eval_loss:.4f}) "
                        fold_patience_counter = 0
                    elif new_qwk > fold_best_qwk:
                        fold_best_qwk = new_qwk
                        save_flag = True
                        improvement_msg += f"QWK 개선 ({new_qwk:.4f}) "
                        fold_patience_counter = 0
                    else:
                        fold_patience_counter += 1
                        improvement_msg = f"성능 개선 없음 (patience: {fold_patience_counter}/{patience})"
                    
                    if save_flag:
                        if not os.path.exists(model_save_dir):
                            os.makedirs(model_save_dir)
                        self.bert_regression_by_word_document.save_pretrained(doc_model_save_dir)
                        self.bert_regression_by_chunk.save_pretrained(chunk_model_save_dir)
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
            
            fold_time = time.time() - fold_start_time
            log_message(f"=== Fold {fold + 1} 완료 (총 소요시간: {fold_time:.2f}초) ===\n")
        
        # 전체 학습 완료 후 결과 저장
        os.makedirs('./train_valid_loss', exist_ok=True)
        np.save(f'./train_valid_loss/{mode}_loss.npy', np.array(loss_list))
        np.save(f'./train_valid_loss/{mode}_mse.npy', np.array(mse_list))
        np.save(f'./train_valid_loss/{mode}_mae.npy', np.array(mae_list))
        
        final_mse = np.mean(mse_list)
        final_mae = np.mean(mae_list)
        
        log_message("="*80)
        log_message("학습 완료!")
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
            
            # 결합
            combined_predictions = (predictions_sentence + predictions_full) / 2
            predictions = combined_predictions.cpu().numpy()[0]  # (11,)
        
        # 결과를 딕셔너리로 반환
        result_dict = {}
        for i, criterion in enumerate(self.criterion_names):
            result_dict[criterion] = float(predictions[i])
        
        return result_dict, predictions
