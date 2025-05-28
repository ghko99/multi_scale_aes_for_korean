"""
한국어 에세이 데이터셋을 위한 11개 평가 기준 멀티-태스크 회귀 모델 사용 예제
"""

import pandas as pd
import numpy as np
from document_bert_architectures import DocumentBertSentenceChunkAttentionLSTM, DocumentBertCombineWordDocumentLinear
from encoder import encode_documents_by_sentence, encode_documents_full_text
from evaluate import evaluation_multi_regression, print_evaluation_results, plot_evaluation_results, analyze_prediction_errors
import torch

# GPU 사용 가능 여부 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"사용 디바이스: {device}")

# 11개 평가 기준 정의
CRITERION_NAMES = [
            '문법 정확도', '단어 선택의 적절성', '문장 표현', '문단 내 구조', '문단 간 구조',
            '구조의 일관성', '분량의 적절성', '주제 명료성', '창의성', 
            '프롬프트 독해력', '설명의 구체성'
]

def create_sample_multi_data():
    """멀티-태스크 회귀를 위한 샘플 데이터 생성"""
    essays = [
        "환경 보호의 중요성에 대해 논의해보겠습니다.#@문장구분#지구 온난화는 현재 인류가 직면한 가장 심각한 문제 중 하나입니다.#@문장구분#이산화탄소 배출량을 줄이기 위한 구체적인 방안이 필요합니다.#@문장구분#개인과 기업, 정부가 모두 협력해야 합니다.",
        
        "교육 시스템의 혁신이 필요합니다.#@문장구분#21세기 교육은 단순한 지식 전달을 넘어서야 합니다.#@문장구분#창의적 사고와 문제 해결 능력을 기르는 것이 중요합니다.#@문장구분#기술을 활용한 맞춤형 교육이 해답이 될 수 있습니다.",
        
        "인공지능의 발달이 사회에 미치는 영향을 살펴보겠습니다.#@문장구분#AI 기술은 우리 생활의 많은 부분을 변화시키고 있습니다.#@문장구분#일자리 변화와 새로운 기회 창출이 동시에 일어나고 있습니다.#@문장구분#윤리적 고려사항도 함께 논의되어야 합니다.",
        
        "건강한 라이프스타일의 중요성에 대해 말씀드리겠습니다.#@문장구분#규칙적인 운동과 균형 잡힌 식단이 기본입니다.#@문장구분#정신 건강도 신체 건강만큼 중요합니다.#@문장구분#스트레스 관리와 충분한 휴식이 필요합니다.",
        
        "문화 다양성과 상호 이해에 대해 생각해봅시다.#@문장구분#서로 다른 문화적 배경을 가진 사람들과의 소통이 중요합니다.#@문장구분#편견을 버리고 열린 마음으로 접근해야 합니다.#@문장구분#다양성이 사회를 더욱 풍요롭게 만들어줍니다.",
        
        "경제적 불평등 해소 방안을 모색해봅시다.#@문장구분#소득 격차가 점점 벌어지고 있는 현실입니다.#@문장구분#교육 기회의 평등과 공정한 경쟁이 필요합니다.#@문장구분#사회 안전망 구축이 시급한 과제입니다."
    ]
    
    # 각 에세이에 대해 11개 평가 기준의 점수 (0~3 범위)
    # 논리성, 창의성, 설득력, 근거의풍부함, 구성의완성도, 표현의정확성, 어휘의다양성, 문법의정확성, 내용의깊이, 독창성, 전체적일관성
    labels = [
        [2.5, 2.0, 2.3, 2.1, 2.4, 2.2, 2.0, 2.3, 2.2, 1.8, 2.3],  # 환경 보호 에세이
        [2.8, 2.5, 2.6, 2.4, 2.7, 2.5, 2.3, 2.4, 2.6, 2.2, 2.5],  # 교육 혁신 에세이
        [2.3, 2.8, 2.4, 2.2, 2.1, 2.0, 2.4, 2.1, 2.5, 2.6, 2.2],  # AI 영향 에세이
        [2.0, 1.8, 2.1, 2.0, 2.2, 2.3, 1.9, 2.2, 2.0, 1.7, 2.1],  # 건강 라이프스타일 에세이
        [2.4, 2.3, 2.5, 2.2, 2.3, 2.4, 2.1, 2.3, 2.3, 2.0, 2.4],  # 문화 다양성 에세이
        [2.6, 2.1, 2.7, 2.5, 2.4, 2.2, 2.2, 2.1, 2.4, 2.0, 2.3]   # 경제 불평등 에세이
    ]
    
    return essays, labels

# 1. 데이터 준비
print("=== 멀티-태스크 회귀 데이터 준비 ===")
essays, labels = create_sample_multi_data()
print(f"에세이 수: {len(essays)}")
print(f"라벨 형태: {np.array(labels).shape}")  # (N, 11)
print(f"평가 기준: {CRITERION_NAMES}")
print(f"\n첫 번째 에세이 점수:")
for i, criterion in enumerate(CRITERION_NAMES):
    print(f"  {criterion}: {labels[0][i]:.1f}점")

# DataFrame으로 변환
df_essays = pd.DataFrame({'essay': essays})
df_labels = pd.DataFrame(labels, columns=CRITERION_NAMES)

print(f"\n라벨 통계:")
print(df_labels.describe())

# 2. 모델 초기화
print("\n=== 모델 초기화 ===")
from model_architecture_bert_multi_scale_multi_loss import DocumentBertScoringModel

class Args:
    def __init__(self):
        self.device = device
        self.batch_size = 2
        self.model_directory = './models'
        self.result_file = 'multi_regression_result.txt'

args = Args()
model = DocumentBertScoringModel(load_model=False, args=args)
print("멀티-태스크 회귀 모델 초기화 완료")

# 3. 데이터 인코딩 테스트  
print("\n=== 데이터 인코딩 테스트 ===")
sentence_encoded, sentence_lengths = encode_documents_by_sentence(
    essays, model.bert_tokenizer, max_input_length=128)
print(f"문장별 인코딩 결과: {sentence_encoded.shape}")
print(f"각 문서의 문장 수: {sentence_lengths.tolist()}")

full_encoded, full_lengths = encode_documents_full_text(
    essays, model.bert_tokenizer, max_input_length=512)
print(f"전체 문서 인코딩 결과: {full_encoded.shape}")

# 4. 단일 예측 테스트
print("\n=== 단일 예측 테스트 ===")
test_essay = "디지털 시대의 교육 혁신이 필요합니다.#@문장구분#온라인 학습 플랫폼이 발달하고 있습니다.#@문장구분#개인 맞춤형 교육이 가능해졌습니다.#@문장구분#하지만 디지털 격차 문제도 고려해야 합니다."

result_dict, predictions = model.predict_single(test_essay)
print("예측 결과:")
for criterion, score in result_dict.items():
    print(f"  {criterion}: {score:.3f}점")

# 5. 평가 함수 테스트
print("\n=== 평가 함수 테스트 ===")
# 가상의 예측 결과 생성 (실제로는 모델에서 나옴)
true_labels_array = np.array(labels)
# 실제 값에 약간의 노이즈를 추가해서 가상 예측값 생성
pred_labels_array = true_labels_array + np.random.normal(0, 0.2, true_labels_array.shape)
pred_labels_array = np.clip(pred_labels_array, 0, 3)  # 0-3 범위로 클리핑

# 평가 수행
results = evaluation_multi_regression(true_labels_array, pred_labels_array, CRITERION_NAMES)
print_evaluation_results(results, detailed=True)

# 6. 시각화 (matplotlib 필요)
print("\n=== 결과 시각화 ===")
try:
    plot_evaluation_results(results, save_path='evaluation_results.png')
    print("시각화 완료: evaluation_results.png 저장됨")
except ImportError:
    print("matplotlib가 설치되지 않아 시각화를 건너뜁니다.")

# 7. 오차 분석
print("\n=== 예측 오차 분석 ===")
analyze_prediction_errors(true_labels_array, pred_labels_array, CRITERION_NAMES, top_k=3)

# 8. 실제 학습 예제
def train_model_example():
    """실제 데이터셋으로 모델 학습하는 예제"""
    print("\n=== 모델 학습 예제 ===")
    
    # 실제 사용 시에는 더 많은 데이터가 필요합니다
    # 학습 데이터 준비
    train_data = (df_essays['essay'], df_labels)
    
    print("학습 데이터 형태:")
    print(f"  에세이: {len(train_data[0])}")
    print(f"  라벨: {train_data[1].shape}")
    
    # 학습 실행 (실제로는 더 많은 데이터와 에포크 필요)
    print("\n실제 학습을 위해서는 다음과 같이 호출:")
    print("model.fit(train_data, test=test_data, mode='korean_multi_regression')")
    
    # 주석 해제하여 실제 학습 실행 가능
    model.fit(train_data, test=train_data, mode='korean_multi_regression')

train_model_example()


# 9. 모델 저장 및 로드 예제
def save_load_model_example():
    """모델 저장 및 로드 예제"""
    print("\n=== 모델 저장 및 로드 예제 ===")
    
    # 모델 저장
    save_dir_word = './saved_models/multi_regression_word_doc_model'
    save_dir_chunk = './saved_models/multi_regression_chunk_model'
    
    print("모델 저장:")
    print(f"  Word Document Model: {save_dir_word}")
    print(f"  Chunk Model: {save_dir_chunk}")
    
    # 실제 저장 (주석 해제하여 사용)
    # model.bert_regression_by_word_document.save_pretrained(save_dir_word)
    # model.bert_regression_by_chunk.save_pretrained(save_dir_chunk)
    
    # 저장된 모델 로드
    print("\n저장된 모델 로드:")
    print("""
    loaded_model = DocumentBertScoringModel(
        load_model=True,
        word_doc_model_path='./saved_models/multi_regression_word_doc_model',
        chunk_model_path='./saved_models/multi_regression_chunk_model',
        args=args
    )
    """)

# 10. 배치 예측 함수
def batch_predict_multi(model, essays, batch_size=4):
    """여러 에세이에 대한 배치 예측 (11개 평가 기준)"""
    all_predictions = []
    
    for i in range(0, len(essays), batch_size):
        batch_essays = essays[i:i+batch_size]
        batch_predictions = []
        
        for essay in batch_essays:
            _, predictions = model.predict_single(essay)
            batch_predictions.append(predictions)
        
        all_predictions.extend(batch_predictions)
    
    return np.array(all_predictions)

# 11. 데이터 전처리 함수
def preprocess_multi_regression_data(essays, labels_matrix):
    """멀티-태스크 회귀를 위한 데이터 전처리"""
    processed_essays = []
    processed_labels = []
    
    for essay, label_row in zip(essays, labels_matrix):
        # 에세이 전처리
        essay = essay.strip()
        if "#@문장구분#" not in essay:
            # 간단한 문장 분할 (실제로는 더 정교한 분할 필요)
            sentences = essay.split('. ')
            essay = "#@문장구분#".join(sentences)
        
        # 라벨 검증 (0~3 범위, 11개)
        label_row = np.array(label_row)
        if len(label_row) != 11:
            print(f"경고: 라벨 개수가 11개가 아닙니다. ({len(label_row)}개)")
            continue
        
        # 0~3 범위로 클리핑
        label_row = np.clip(label_row, 0.0, 3.0)
        
        processed_essays.append(essay)
        processed_labels.append(label_row.tolist())
    
    return processed_essays, processed_labels

# 12. CSV 파일 처리 예제
def load_csv_multi_regression_example():
    """CSV 파일에서 멀티-태스크 회귀 데이터 로드 예제"""
    print("\n=== CSV 파일 로드 예제 ===")
    
    # 예상되는 CSV 파일 형태
    sample_csv_structure = """
    예상 CSV 구조:
    essay_text,논리성,창의성,설득력,근거의풍부함,구성의완성도,표현의정확성,어휘의다양성,문법의정확성,내용의깊이,독창성,전체적일관성
    "안녕하세요.#@문장구분#오늘은...",2.5,2.0,2.3,2.1,2.4,2.2,2.0,2.3,2.2,1.8,2.3
    "교육의 중요성에...",2.8,2.5,2.6,2.4,2.7,2.5,2.3,2.4,2.6,2.2,2.5
    ...
    """
    print(sample_csv_structure)
    
    # 실제 CSV 로드 코드 예제
    load_code_example = """
    # 실제 CSV 파일 로드 방법:
    df = pd.read_csv('korean_essays_multi_regression.csv')
    
    essays = df['essay_text'].tolist()
    label_columns = ['논리성', '창의성', '설득력', '근거의풍부함', '구성의완성도',
                    '표현의정확성', '어휘의다양성', '문법의정확성', '내용의깊이', 
                    '독창성', '전체적일관성']
    labels = df[label_columns].values  # (N, 11) numpy array
    
    # 데이터 전처리
    processed_essays, processed_labels = preprocess_multi_regression_data(essays, labels)
    
    # DataFrame으로 변환
    df_essays = pd.Series(processed_essays)
    df_labels = pd.DataFrame(processed_labels, columns=label_columns)
    
    # 모델 학습
    model = DocumentBertScoringModel(load_model=False, args=args)
    model.fit((df_essays, df_labels), mode='korean_multi_regression')
    """
    print(load_code_example)

# 13. 모델 성능 모니터링
def monitor_training_progress():
    """학습 진행 상황 모니터링 예제"""
    print("\n=== 학습 진행 모니터링 ===")
    
    monitoring_code = """
    # 학습 중 손실 및 메트릭 모니터링
    import matplotlib.pyplot as plt
    
    # 저장된 학습 결과 로드
    train_losses = np.load('./train_valid_loss/korean_multi_regression_loss.npy')
    train_mse = np.load('./train_valid_loss/korean_multi_regression_mse.npy')
    train_mae = np.load('./train_valid_loss/korean_multi_regression_mae.npy')
    
    # 학습 곡선 그리기
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_mse)
    plt.title('Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    
    plt.subplot(1, 3, 3)
    plt.plot(train_mae)
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    """
    print(monitoring_code)

# 실행 예제들
if __name__ == "__main__":
    # 전처리 예제
    processed_essays, processed_labels = preprocess_multi_regression_data(essays, labels)
    print(f"\n전처리된 데이터: 에세이 {len(processed_essays)}개, 라벨 형태 {np.array(processed_labels).shape}")
    
    # 배치 예측 예제
    print("\n=== 배치 예측 테스트 ===")
    batch_predictions = batch_predict_multi(model, processed_essays[:3])
    print(f"배치 예측 결과 형태: {batch_predictions.shape}")
    print("첫 번째 에세이 예측 점수:")
    for i, criterion in enumerate(CRITERION_NAMES):
        print(f"  {criterion}: {batch_predictions[0][i]:.3f}점")
    
    # 다른 예제들 실행
    train_model_example()
    save_load_model_example()
    load_csv_multi_regression_example()
    monitor_training_progress()
    
    print("\n" + "="*60)
    print("멀티-태스크 회귀 모델 사용 가이드")
    print("="*60)
    
    print("""
📋 주요 특징:
  • 하나의 에세이 → 11개 평가 기준 점수 (0~3점)
  • 문장별 분할 처리 (LSTM + Attention)
  • 전체 문서 처리 (Linear)
  • 두 모델의 앙상블

📊 평가 기준 (11개):
  1. 논리성        2. 창의성        3. 설득력
  4. 근거의풍부함   5. 구성의완성도   6. 표현의정확성  
  7. 어휘의다양성   8. 문법의정확성   9. 내용의깊이
  10. 독창성      11. 전체적일관성

🔧 사용 방법:
  1. 데이터 준비: CSV 파일 (essay_text + 11개 점수 컬럼)
  2. 모델 초기화: DocumentBertScoringModel()
  3. 학습: model.fit(train_data)
  4. 예측: model.predict_single(essay) 또는 batch_predict_multi()
  5. 평가: evaluation_multi_regression()

💡 주의사항:
  • 문장 구분자: "#@문장구분#" 사용
  • 점수 범위: 0~3점 (실수)
  • 배치 크기: GPU 메모리에 따라 조정
  • 학습 데이터: 충분한 양 필요 (수천~수만 개)

📈 성능 지표:
  • MSE, MAE, RMSE: 회귀 성능
  • R², Pearson: 상관관계
  • 허용 오차 정확도: ±0.1, ±0.2, ±0.5
  • QWK: 순서형 평가에 적합
    """)
    
    print("\n사용 예제 완료! 🎉")
    print("실제 데이터로 학습할 준비가 되었습니다.")
