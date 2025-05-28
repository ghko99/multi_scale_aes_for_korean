import pandas as pd
import torch
import numpy as np
from model_architecture_bert_multi_scale_multi_loss import DocumentBertScoringModel
import warnings
import os
from datetime import datetime

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

def analyze_data_distribution(train_labels, test_labels, criterion_names):
    """데이터 분포 분석"""
    print("\n" + "="*60)
    print("데이터 분포 분석")
    print("="*60)
    
    train_array = np.array(train_labels)
    test_array = np.array(test_labels)
    
    print(f"훈련 데이터 형태: {train_array.shape}")
    print(f"검증 데이터 형태: {test_array.shape}")
    
    print("\n평가 기준별 점수 분포:")
    print("-" * 80)
    print(f"{'평가기준':<12} {'훈련평균':<10} {'훈련표준편차':<12} {'검증평균':<10} {'검증표준편차':<12} {'범위':<15}")
    print("-" * 80)
    
    for i, criterion in enumerate(criterion_names):
        train_mean = np.mean(train_array[:, i])
        train_std = np.std(train_array[:, i])
        test_mean = np.mean(test_array[:, i])
        test_std = np.std(test_array[:, i])
        score_range = f"{np.min(train_array[:, i]):.1f}-{np.max(train_array[:, i]):.1f}"
        
        print(f"{criterion:<12} {train_mean:<10.3f} {train_std:<12.3f} {test_mean:<10.3f} {test_std:<12.3f} {score_range:<15}")
    
    # 전체 통계
    print("\n전체 통계:")
    print(f"전체 평균 점수: 훈련={np.mean(train_array):.3f}, 검증={np.mean(test_array):.3f}")
    print(f"전체 표준편차: 훈련={np.std(train_array):.3f}, 검증={np.std(test_array):.3f}")

def load_data():
    """데이터 로드 및 전처리"""
    print("데이터 로드 중...")
    
    try:
        train = pd.read_csv('./data/train.csv', encoding='utf-8-sig')
        test = pd.read_csv('./data/valid.csv', encoding='utf-8-sig')
    except FileNotFoundError as e:
        print(f"데이터 파일을 찾을 수 없습니다: {e}")
        print("data/ 폴더에 train.csv와 valid.csv 파일이 있는지 확인해주세요.")
        return None, None
    
    print(f"훈련 데이터: {len(train)}개")
    print(f"검증 데이터: {len(test)}개")

    train_essays = train['essay'].to_list()
    train_score_avg = train['essay_score_avg'].to_list()

    test_essays = test['essay'].to_list()
    test_score_avg = test['essay_score_avg'].to_list()

    train_labels = []
    test_labels = []

    # 라벨 파싱 및 검증
    print("라벨 데이터 파싱 중...")
    
    for i, score_str in enumerate(train_score_avg):
        try:
            train_scores = score_str.split('#')
            train_scores = [float(score) for score in train_scores]
            if len(train_scores) != 11:
                print(f"경고: 훈련 샘플 {i}의 점수 개수가 {len(train_scores)}개입니다. (기대값: 11)")
            train_labels.append(train_scores)
        except Exception as e:
            print(f"훈련 데이터 {i} 파싱 오류: {e}")
            return None, None

    for i, score_str in enumerate(test_score_avg):
        try:
            test_scores = score_str.split('#')
            test_scores = [float(score) for score in test_scores]
            if len(test_scores) != 11:
                print(f"경고: 검증 샘플 {i}의 점수 개수가 {len(test_scores)}개입니다. (기대값: 11)")
            test_labels.append(test_scores)
        except Exception as e:
            print(f"검증 데이터 {i} 파싱 오류: {e}")
            return None, None
    
    print("데이터 로드 완료!")
    return (train_essays, train_labels), (test_essays, test_labels)

def create_directories():
    """필요한 디렉토리 생성"""
    directories = ['./models', './logs', './train_valid_loss', './results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    print("="*60)
    print("개선된 한국어 에세이 자동 채점 시스템")
    print("="*60)
    
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 디바이스: {device}")
    
    if device == 'cuda':
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    
    # 평가 기준 정의
    CRITERION_NAMES = [
        '문법 정확도', '단어 선택의 적절성', '문장 표현', '문단 내 구조', '문단 간 구조',
        '구조의 일관성', '분량의 적절성', '주제 명료성', '창의성', 
        '프롬프트 독해력', '설명의 구체성'
    ]
    
    # 필요한 디렉토리 생성
    create_directories()
    
    # 데이터 로드
    data_result = load_data()
    if data_result is None:
        return
    
    (train_essays, train_labels), (test_essays, test_labels) = data_result
    
    # 데이터 분포 분석
    analyze_data_distribution(train_labels, test_labels, CRITERION_NAMES)
    
    # 모델 초기화
    print("\n모델 초기화 중...")
    
    # 개선된 설정으로 모델 생성
    args = {
        'device': device,
        'batch_size': 8 if device == 'cuda' else 4,  # GPU 메모리에 따라 조정
        'gradient_accumulation_steps': 4 if device == 'cuda' else 2,
        'max_grad_norm': 1.0,
        'warmup_ratio': 0.1,
        'label_smoothing': 0.1
    }
    
    # 배치 크기를 GPU 메모리에 따라 동적 조정
    if device == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory
        if total_memory < 8 * 1024**3:  # 8GB 미만
            args['batch_size'] = 4
            args['gradient_accumulation_steps'] = 2
        elif total_memory < 12 * 1024**3:  # 12GB 미만
            args['batch_size'] = 6
            args['gradient_accumulation_steps'] = 3
        else:  # 12GB 이상
            args['batch_size'] = 8
            args['gradient_accumulation_steps'] = 4
    
    print(f"최종 배치 설정: batch_size={args['batch_size']}, accumulation_steps={args['gradient_accumulation_steps']}")
    
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    model = DocumentBertScoringModel(load_model=False, args=Args(**args))
    
    # 데이터 준비
    df_train_essays = pd.DataFrame({'essay': train_essays})
    df_train_labels = pd.DataFrame(train_labels, columns=CRITERION_NAMES)

    df_test_essays = pd.DataFrame({'essay': test_essays})
    df_test_labels = pd.DataFrame(test_labels, columns=CRITERION_NAMES)

    train_data = (df_train_essays['essay'], df_train_labels)
    test_data = (df_test_essays['essay'], df_test_labels)
    
    # 학습 시작
    print(f"\n학습 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # 개선된 학습 실행
        model.fit(
            train_data, 
            test=test_data, 
            mode=f'improved_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            patience=8,  # 인내심 증가
            log_dir='./logs'
        )
        
        print("\n학습 완료!")
        
        # 최종 평가
        print("\n최종 모델 평가 중...")
        model.load_best_model()
        
        final_mse, final_mae, (true_labels, pred_labels), final_qwk, final_loss, _, _, _ = model.predict_for_regress(
            test_data, writeflag=True
        )
        
        print(f"\n최종 성능:")
        print(f"MSE: {final_mse:.4f}")
        print(f"MAE: {final_mae:.4f}")
        print(f"QWK: {final_qwk:.4f}")
        print(f"Loss: {final_loss:.4f}")
        
        # 학습 진행 상황 분석
        print("\n학습 진행 상황 분석 중...")
        model.analyze_training_progress()
        
        # 샘플 예측 테스트
        print("\n샘플 예측 테스트:")
        if len(test_essays) > 0:
            sample_essay = test_essays[0]
            sample_true = test_labels[0]
            
            result_dict, predictions = model.predict_single(sample_essay)
            
            print(f"에세이 미리보기: {sample_essay[:100]}...")
            print("\n예측 결과:")
            for i, criterion in enumerate(CRITERION_NAMES):
                print(f"  {criterion}: 실제={sample_true[i]:.3f}, 예측={predictions[i]:.3f}, "
                      f"오차={abs(sample_true[i] - predictions[i]):.3f}")
        
        print(f"\n전체 실행 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 학습이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # GPU 메모리 정리
        if device == 'cuda':
            torch.cuda.empty_cache()
            print("GPU 메모리 정리 완료")

if __name__ == "__main__":
    main()