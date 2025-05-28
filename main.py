import pandas as pd
import torch
import numpy as np
from model_architecture_bert_multi_scale_multi_loss import DocumentBertScoringModel
import warnings
import os
from datetime import datetime
import time

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

def analyze_data_distribution(train_labels, test_labels, criterion_names):
    """데이터 분포 분석"""
    print("\n" + "="*60)
    print("📊 데이터 분포 분석")
    print("="*60)
    
    train_array = np.array(train_labels)
    test_array = np.array(test_labels)
    
    print(f"훈련 데이터 형태: {train_array.shape}")
    print(f"검증 데이터 형태: {test_array.shape}")
    
    print("\n평가 기준별 점수 분포:")
    print("-" * 90)
    print(f"{'평가기준':<15} {'훈련평균':<8} {'훈련표준편차':<10} {'검증평균':<8} {'검증표준편차':<10} {'난이도':<8}")
    print("-" * 90)
    
    difficulty_analysis = []
    
    for i, criterion in enumerate(criterion_names):
        train_mean = np.mean(train_array[:, i])
        train_std = np.std(train_array[:, i])
        test_mean = np.mean(test_array[:, i])
        test_std = np.std(test_array[:, i])
        
        # 난이도 분류
        if train_mean >= 2.5 and train_std <= 0.5:
            difficulty = "🟢 쉬움"
        elif train_mean <= 1.5 or train_std >= 1.2:
            difficulty = "🔴 어려움"
        else:
            difficulty = "🟡 보통"
        
        difficulty_analysis.append((criterion, difficulty, train_mean, train_std))
        print(f"{criterion:<15} {train_mean:<8.3f} {train_std:<10.3f} {test_mean:<8.3f} {test_std:<10.3f} {difficulty:<8}")
    
    # 전체 통계
    print(f"\n전체 통계:")
    print(f"전체 평균 점수: 훈련={np.mean(train_array):.3f}, 검증={np.mean(test_array):.3f}")
    print(f"전체 표준편차: 훈련={np.std(train_array):.3f}, 검증={np.std(test_array):.3f}")
    
    # 불균형 정도 분석
    easy_count = sum(1 for _, diff, _, _ in difficulty_analysis if "쉬움" in diff)
    medium_count = sum(1 for _, diff, _, _ in difficulty_analysis if "보통" in diff)
    hard_count = sum(1 for _, diff, _, _ in difficulty_analysis if "어려움" in diff)
    
    print(f"\n🎯 난이도 분포:")
    print(f"  쉬운 기준: {easy_count}개")
    print(f"  보통 기준: {medium_count}개")
    print(f"  어려운 기준: {hard_count}개")
    
    if hard_count >= 3:
        print(f"  ⚠️  심각한 불균형 감지! 특별한 학습 전략이 필요합니다.")
    elif hard_count >= 2:
        print(f"  ⚠️  불균형 감지! 균형 잡힌 학습을 적용합니다.")
    else:
        print(f"  ✅ 비교적 균형적인 데이터입니다.")
    
    return difficulty_analysis

def load_data():
    """데이터 로드 및 전처리"""
    print("📁 데이터 로드 중...")
    start_time = time.time()
    
    try:
        train = pd.read_csv('./data/train.csv', encoding='utf-8-sig')
        test = pd.read_csv('./data/valid.csv', encoding='utf-8-sig')
    except FileNotFoundError as e:
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {e}")
        print("data/ 폴더에 train.csv와 valid.csv 파일이 있는지 확인해주세요.")
        return None, None
    
    print(f"✅ 파일 로드 완료: 훈련 {len(train)}개, 검증 {len(test)}개")

    train_essays = train['essay'].to_list()
    train_score_avg = train['essay_score_avg'].to_list()
    test_essays = test['essay'].to_list()
    test_score_avg = test['essay_score_avg'].to_list()

    train_labels = []
    test_labels = []

    # 라벨 파싱 및 검증
    print("🔍 라벨 데이터 파싱 중...")
    
    for i, score_str in enumerate(train_score_avg):
        try:
            train_scores = score_str.split('#')
            train_scores = [float(score) for score in train_scores]
            if len(train_scores) != 11:
                print(f"⚠️  훈련 샘플 {i}: 점수 개수 {len(train_scores)}개 (기대값: 11)")
            # 0-3 범위 검증
            train_scores = [max(0.0, min(3.0, score)) for score in train_scores]
            train_labels.append(train_scores)
        except Exception as e:
            print(f"❌ 훈련 데이터 {i} 파싱 오류: {e}")
            return None, None

    for i, score_str in enumerate(test_score_avg):
        try:
            test_scores = score_str.split('#')
            test_scores = [float(score) for score in test_scores]
            if len(test_scores) != 11:
                print(f"⚠️  검증 샘플 {i}: 점수 개수 {len(test_scores)}개 (기대값: 11)")
            # 0-3 범위 검증
            test_scores = [max(0.0, min(3.0, score)) for score in test_scores]
            test_labels.append(test_scores)
        except Exception as e:
            print(f"❌ 검증 데이터 {i} 파싱 오류: {e}")
            return None, None
    
    load_time = time.time() - start_time
    print(f"✅ 데이터 로드 완료! (소요시간: {load_time:.2f}초)")
    return (train_essays, train_labels), (test_essays, test_labels)

def create_directories():
    """필요한 디렉토리 생성"""
    directories = ['./models', './logs', './train_valid_loss', './results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_optimal_settings(device):
    """GPU 메모리에 따른 최적 설정"""
    if device == 'cpu':
        return {
            'batch_size': 4,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0
        }
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    gpu_gb = total_memory // 1024**3
    
    if gpu_gb >= 24:  # 24GB+
        return {
            'batch_size': 16,
            'gradient_accumulation_steps': 4,
            'max_grad_norm': 1.0
        }
    elif gpu_gb >= 16:  # 16GB+
        return {
            'batch_size': 12,
            'gradient_accumulation_steps': 3,
            'max_grad_norm': 1.0
        }
    elif gpu_gb >= 12:  # 12GB+
        return {
            'batch_size': 10,
            'gradient_accumulation_steps': 2,
            'max_grad_norm': 1.0
        }
    elif gpu_gb >= 8:   # 8GB+
        return {
            'batch_size': 8,
            'gradient_accumulation_steps': 2,
            'max_grad_norm': 1.0
        }
    else:  # 8GB 미만
        return {
            'batch_size': 6,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 0.5
        }

def main():
    print("="*80)
    print("🚀 수정된 한국어 에세이 자동 채점 시스템 (불균형 데이터 대응)")
    print("="*80)
    
    start_time = time.time()
    
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 사용 디바이스: {device}")
    
    if device == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        gpu_name = torch.cuda.get_device_name(0)
        print(f"💾 GPU: {gpu_name} ({gpu_memory}GB)")
        
        # GPU 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
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
    difficulty_analysis = analyze_data_distribution(train_labels, test_labels, CRITERION_NAMES)
    
    # 최적 설정 결정
    optimal_settings = get_optimal_settings(device)
    print(f"\n⚙️  최적 설정:")
    print(f"   배치 크기: {optimal_settings['batch_size']}")
    print(f"   그래디언트 누적: {optimal_settings['gradient_accumulation_steps']}")
    print(f"   그래디언트 클리핑: {optimal_settings['max_grad_norm']}")
    
    # 모델 초기화
    print(f"\n🤖 모델 초기화 중...")
    model_start = time.time()
    
    # 개선된 설정으로 모델 생성
    args = {
        'device': device,
        'batch_size': optimal_settings['batch_size'],
        'gradient_accumulation_steps': optimal_settings['gradient_accumulation_steps'],
        'max_grad_norm': optimal_settings['max_grad_norm'],
        'warmup_ratio': 0.1,
        'label_smoothing': 0.1
    }
    
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    model = DocumentBertScoringModel(load_model=False, args=Args(**args))
    
    model_init_time = time.time() - model_start
    print(f"✅ 모델 초기화 완료! (소요시간: {model_init_time:.2f}초)")
    
    # 데이터 준비
    df_train_essays = pd.DataFrame({'essay': train_essays})
    df_train_labels = pd.DataFrame(train_labels, columns=CRITERION_NAMES)
    df_test_essays = pd.DataFrame({'essay': test_essays})
    df_test_labels = pd.DataFrame(test_labels, columns=CRITERION_NAMES)

    train_data = (df_train_essays['essay'], df_train_labels)
    test_data = (df_test_essays['essay'], df_test_labels)
    
    # 불균형 분석
    model.analyze_data_imbalance(train_labels)
    
    # 학습 시작
    print(f"\n🎓 학습 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        # 수정된 학습 실행
        train_start = time.time()
        
        model.fit(
            train_data, 
            test=test_data, 
            mode=f'fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            patience=6,  # 조금 더 빠른 early stopping
            log_dir='./logs'
        )
        
        train_time = time.time() - train_start
        print(f"\n✅ 학습 완료! (총 학습 시간: {train_time/60:.1f}분)")
        
        # 최종 평가
        print(f"\n📊 최종 모델 평가 중...")
        eval_start = time.time()
        
        final_mse, final_mae, (true_labels, pred_labels), final_qwk, final_loss, criterion_mse, criterion_mae, criterion_qwk = model.predict_for_regress(
            test_data, writeflag=True
        )
        
        eval_time = time.time() - eval_start
        
        print(f"\n🎯 최종 성능 (평가시간: {eval_time:.2f}초):")
        print(f"   전체 MSE: {final_mse:.4f}")
        print(f"   전체 MAE: {final_mae:.4f}")
        print(f"   전체 QWK: {final_qwk:.4f}")
        print(f"   평가 손실: {final_loss:.4f}")
        
        # 난이도별 성능 분석
        print(f"\n📈 난이도별 성능 분석:")
        
        easy_criteria = [i for i, (_, diff, _, _) in enumerate(difficulty_analysis) if "쉬움" in diff]
        medium_criteria = [i for i, (_, diff, _, _) in enumerate(difficulty_analysis) if "보통" in diff]
        hard_criteria = [i for i, (_, diff, _, _) in enumerate(difficulty_analysis) if "어려움" in diff]
        
        if easy_criteria:
            easy_qwk = np.mean([criterion_qwk[i] for i in easy_criteria])
            print(f"   🟢 쉬운 기준 평균 QWK: {easy_qwk:.4f}")
        
        if medium_criteria:
            medium_qwk = np.mean([criterion_qwk[i] for i in medium_criteria])
            print(f"   🟡 보통 기준 평균 QWK: {medium_qwk:.4f}")
        
        if hard_criteria:
            hard_qwk = np.mean([criterion_qwk[i] for i in hard_criteria])
            print(f"   🔴 어려운 기준 평균 QWK: {hard_qwk:.4f}")
            
            print(f"\n   어려운 기준 상세:")
            for i in hard_criteria:
                criterion_name = CRITERION_NAMES[i]
                print(f"     {criterion_name}: MSE={criterion_mse[i]:.4f}, QWK={criterion_qwk[i]:.4f}")
        
        # 성능 개선 제안
        print(f"\n💡 성능 개선 제안:")
        if final_qwk < 0.7:
            print("   - 더 많은 학습 데이터 확보")
            print("   - 데이터 증강 기법 적용")
            if hard_criteria:
                print("   - 어려운 기준 전용 모델 고려")
        elif final_qwk < 0.8:
            print("   - 앙상블 모델 추가")
            print("   - 하이퍼파라미터 튜닝")
        else:
            print("   - 현재 성능이 우수합니다! 🎉")
        
        # 샘플 예측 테스트
        print(f"\n🔍 샘플 예측 테스트:")
        if len(test_essays) > 0:
            sample_idx = 0
            sample_essay = test_essays[sample_idx]
            sample_true = test_labels[sample_idx]
            
            prediction_start = time.time()
            result_dict, predictions = model.predict_single(sample_essay)
            prediction_time = time.time() - prediction_start
            
            print(f"   예측 시간: {prediction_time:.3f}초")
            print(f"   에세이 길이: {len(sample_essay)}자")
            print(f"   에세이 미리보기: {sample_essay[:100]}...")
            
            # 오차가 큰 기준 3개만 출력
            errors = [(CRITERION_NAMES[i], abs(sample_true[i] - predictions[i]), 
                      sample_true[i], predictions[i]) for i in range(11)]
            errors.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n   오차가 큰 상위 3개 기준:")
            for i, (name, error, true_val, pred_val) in enumerate(errors[:3]):
                print(f"     {i+1}. {name}: 실제={true_val:.3f}, 예측={pred_val:.3f}, 오차={error:.3f}")
        
        # 전체 실행 시간
        total_time = time.time() - start_time
        
        print(f"\n🏁 시스템 실행 완료!")
        print(f"   총 실행 시간: {total_time/60:.1f}분")
        print(f"   학습 시간: {train_time/60:.1f}분 ({train_time/total_time*100:.1f}%)")
        print(f"   평가 시간: {eval_time:.1f}초 ({eval_time/total_time*100:.1f}%)")
        
        # 성능 지표 요약
        print(f"\n📋 최종 성능 요약:")
        print(f"   🎯 QWK 점수: {final_qwk:.4f}")
        print(f"   📉 MSE: {final_mse:.4f}")
        print(f"   📊 MAE: {final_mae:.4f}")
        
        # 성능 등급 매기기
        if final_qwk >= 0.85:
            grade = "🥇 우수"
        elif final_qwk >= 0.75:
            grade = "🥈 양호"
        elif final_qwk >= 0.65:
            grade = "🥉 보통"
        else:
            grade = "📈 개선필요"
        
        print(f"   등급: {grade}")
        
        # 메모리 사용량 (CUDA)
        if device == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   💾 GPU 메모리: {memory_used:.1f}GB / {memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)")
        
        # 파일 저장 위치
        print(f"\n📄 생성된 파일:")
        print(f"   💾 모델: ./models/")
        print(f"   📊 로그: ./logs/")
        print(f"   📈 학습곡선: ./train_valid_loss/")
        print(f"   📝 예측결과: ./models/result.txt")
        
        # 다음 단계 제안
        print(f"\n🔜 다음 단계:")
        print("   1. 저장된 모델로 새로운 에세이 채점")
        print("   2. 학습 곡선 시각화로 성능 분석")
        print("   3. 하이퍼파라미터 튜닝으로 성능 개선")
        print("   4. 더 많은 데이터로 재학습")
        
        print(f"\n사용 예시:")
        print("   # 새로운 에세이 채점")
        print("   model.load_best_model()")
        print("   result, scores = model.predict_single('새로운 에세이 내용...')")
        print("   print(result)")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  사용자에 의해 학습이 중단되었습니다.")
        interrupted_time = time.time() - start_time
        print(f"   부분 실행 시간: {interrupted_time/60:.1f}분")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류 해결 제안
        error_str = str(e)
        print(f"\n🔧 오류 해결 제안:")
        
        if "CUDA out of memory" in error_str:
            print("   - 배치 크기를 줄여보세요: batch_size를 4 또는 2로 설정")
            print("   - 그래디언트 누적을 늘려보세요: gradient_accumulation_steps 증가")
            print("   - 문서 길이를 줄여보세요: max_doc_length를 256으로 설정")
        elif "expected Tensor" in error_str:
            print("   - 이미 수정된 버전을 사용하고 있습니다")
            print("   - 데이터 형태를 확인해보세요")
        elif "FileNotFoundError" in error_str:
            print("   - data/ 폴더에 train.csv, valid.csv 파일이 있는지 확인")
            print("   - 파일 경로와 인코딩을 확인해보세요")
        else:
            print("   - GitHub 이슈를 확인하거나 문의해주세요")
            print("   - 로그 파일을 첨부해주시면 도움이 됩니다")
    
    finally:
        # 메모리 정리
        if device == 'cuda':
            torch.cuda.empty_cache()
            print("🧹 GPU 메모리 정리 완료")
        
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n⏰ 시스템 종료: {end_time}")
        print("="*80)

def quick_test():
    """빠른 테스트 모드 (소량 데이터로 동작 확인)"""
    print("🚀 빠른 테스트 모드")
    print("="*50)
    
    # 가상 데이터 생성
    test_essays = [
        "환경 보호는 중요합니다.#@문장구분#우리 모두 노력해야 합니다.#@문장구분#재활용을 실천합시다.",
        "교육의 중요성에 대해 말씀드리겠습니다.#@문장구분#좋은 교육이 미래를 만듭니다.#@문장구분#모든 학생이 공평한 기회를 가져야 합니다.",
        "기술 발전이 사회에 미치는 영향을 살펴보겠습니다.#@문장구분#인공지능이 우리 생활을 바꾸고 있습니다.#@문장구분#하지만 부작용도 있습니다."
    ]
    
    test_labels = [
        [2.5, 2.0, 1.5, 2.2, 1.8, 2.1, 2.3, 2.4, 2.0, 1.6, 2.2],
        [2.8, 2.5, 2.0, 2.6, 2.2, 2.4, 2.7, 2.8, 2.3, 2.1, 2.5],
        [2.3, 2.1, 1.8, 2.0, 1.9, 2.0, 2.2, 2.3, 2.4, 1.9, 2.1]
    ]
    
    print(f"테스트 데이터: {len(test_essays)}개 에세이")
    
    # 간단한 모델 테스트
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args = {
        'device': device,
        'batch_size': 2,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0
    }
    
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    model = DocumentBertScoringModel(load_model=False, args=Args(**args))
    
    # 단일 예측 테스트
    print("\n단일 예측 테스트:")
    result_dict, predictions = model.predict_single(test_essays[0])
    
    print("예측 결과:")
    for criterion, score in result_dict.items():
        print(f"  {criterion}: {score:.3f}")
    
    print("\n✅ 테스트 완료! 모델이 정상 작동합니다.")

if __name__ == "__main__":
    import sys
    
    # 커맨드라인 인자 처리
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            quick_test()
        elif sys.argv[1] == 'help':
            print("사용법:")
            print("  python main_fixed.py        # 전체 학습 실행")
            print("  python main_fixed.py test   # 빠른 테스트")
            print("  python main_fixed.py help   # 도움말")
        else:
            print("알 수 없는 옵션입니다. 'help'를 참조하세요.")
    else:
        main()