from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def quadratic_weighted_kappa_single(rater_a, rater_b, min_rating=None, max_rating=None):
    """단일 기준에 대한 QWK 계산"""
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    
    # 혼동 행렬 생성
    num_ratings = max_rating - min_rating + 1
    conf_mat = np.zeros((num_ratings, num_ratings))
    
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    
    # 가중치 행렬 생성
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
    
    # 기대 행렬 계산
    hist_a = np.sum(conf_mat, axis=1)
    hist_b = np.sum(conf_mat, axis=0)
    expected = np.outer(hist_a, hist_b) / len(rater_a)
    
    # QWK 계산
    numerator = np.sum(weights * conf_mat)
    denominator = np.sum(weights * expected)
    
    if denominator == 0:
        return 1.0
    
    return 1.0 - numerator / denominator



# 기존 함수와의 호환성을 위한 래퍼 함수들
def quadratic_weighted_kappa_multi(rater_a, rater_b):
    """멀티 회귀를 위한 QWK (각 기준별로 계산 후 평균)"""
    rater_a = np.array(rater_a)
    rater_b = np.array(rater_b)
    
    if len(rater_a.shape) == 1:
        return quadratic_weighted_kappa_single(rater_a, rater_b)
    
    qwk_scores = []
    for i in range(rater_a.shape[1]):
        # 0-3 범위를 정수로 변환 (QWK 계산을 위해)
        a_rounded = np.round(rater_a[:, i] * 3).astype(int)  # 0-3 정수
        b_rounded = np.round(rater_b[:, i] * 3).astype(int)  # 0-3 정수
        
        qwk = quadratic_weighted_kappa_single(a_rounded, b_rounded, min_rating=0, max_rating=3)
        qwk_scores.append(qwk)
    
    return np.mean(qwk_scores)



def evaluation_multi_regression(true_labels, pred_labels, criterion_names=None):
    """
    11개 평가 기준에 대한 멀티-태스크 회귀 평가 함수
    
    Args:
        true_labels: (N, 11) 실제 라벨
        pred_labels: (N, 11) 예측 라벨
        criterion_names: 11개 평가 기준 이름 리스트
    
    Returns:
        results: 평가 결과 딕셔너리
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    assert true_labels.shape == pred_labels.shape
    assert true_labels.shape[1] == 11, "11개 평가 기준이어야 합니다"
    
    if criterion_names is None:
        criterion_names = [f'기준_{i+1}' for i in range(11)]
    
    results = {
        'overall': {},
        'by_criterion': {},
        'correlation_matrix': None,
        'score_distribution': {}
    }
    
    # 전체 성능 평가
    overall_mse = mean_squared_error(true_labels.flatten(), pred_labels.flatten())
    overall_mae = mean_absolute_error(true_labels.flatten(), pred_labels.flatten())
    overall_rmse = np.sqrt(overall_mse)
    overall_r2 = r2_score(true_labels.flatten(), pred_labels.flatten())
    overall_pearson = pearsonr(true_labels.flatten(), pred_labels.flatten())[0]
    overall_qwk = quadratic_weighted_kappa_multi(np.round(true_labels).astype(int).flatten() , np.round(pred_labels).astype(int).flatten())

    results['overall'] = {
        'MSE': overall_mse,
        'MAE': overall_mae,
        'RMSE': overall_rmse,
        'R2': overall_r2,
        'Pearson': overall_pearson,
        "QWK" : overall_qwk
    }
    
    # 평가 기준별 성능 평가
    criterion_results = {}
    for i, criterion in enumerate(criterion_names):
        true_criterion = true_labels[:, i]
        pred_criterion = pred_labels[:, i]
        
        mse = mean_squared_error(true_criterion, pred_criterion)
        mae = mean_absolute_error(true_criterion, pred_criterion)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_criterion, pred_criterion)
        pearson = pearsonr(true_criterion, pred_criterion)[0]
        qwk = quadratic_weighted_kappa_multi(np.round(true_criterion).astype(int), np.round(pred_criterion).astype(int))


        # 정확도 (허용 오차 내)
        tolerance_01_acc = np.mean(np.abs(pred_criterion - true_criterion) <= 0.1)
        tolerance_02_acc = np.mean(np.abs(pred_criterion - true_criterion) <= 0.2)
        tolerance_05_acc = np.mean(np.abs(pred_criterion - true_criterion) <= 0.5)
        
        criterion_results[criterion] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Pearson': pearson,
            "QWK" : qwk,
            'Tolerance_0.1_Acc': tolerance_01_acc,
            'Tolerance_0.2_Acc': tolerance_02_acc,
            'Tolerance_0.5_Acc': tolerance_05_acc
        }
    
    results['by_criterion'] = criterion_results
    
    # 평가 기준 간 상관관계 분석
    true_corr_matrix = np.corrcoef(true_labels.T)
    pred_corr_matrix = np.corrcoef(pred_labels.T)
    
    results['correlation_matrix'] = {
        'true_labels': true_corr_matrix,
        'pred_labels': pred_corr_matrix,
        'difference': np.abs(true_corr_matrix - pred_corr_matrix)
    }
    
    # 점수 분포 분석
    for i, criterion in enumerate(criterion_names):
        results['score_distribution'][criterion] = {
            'true_mean': np.mean(true_labels[:, i]),
            'true_std': np.std(true_labels[:, i]),
            'pred_mean': np.mean(pred_labels[:, i]),
            'pred_std': np.std(pred_labels[:, i]),
            'true_range': (np.min(true_labels[:, i]), np.max(true_labels[:, i])),
            'pred_range': (np.min(pred_labels[:, i]), np.max(pred_labels[:, i]))
        }
    
    return results


def print_evaluation_results(results, detailed=True):
    """평가 결과를 보기 좋게 출력"""
    print("=" * 60)
    print("멀티-태스크 회귀 성능 평가 결과")
    print("=" * 60)
    
    # 전체 성능
    print("\n🔍 전체 성능:")
    overall = results['overall']
    print(f"  MSE: {overall['MSE']:.4f}")
    print(f"  MAE: {overall['MAE']:.4f}")
    print(f"  RMSE: {overall['RMSE']:.4f}")
    print(f"  R²: {overall['R2']:.4f}")
    print(f"  Pearson 상관계수: {overall['Pearson']:.4f}")
    print(f"  Quadratic Weighted Kappa Score: {overall['QWK']:.4f}")
    
    # 평가 기준별 성능
    print("\n📊 평가 기준별 성능:")
    print("-" * 100)
    print(f"{'평가기준':<12} {'MSE':<8} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Pearson':<8} {'QWK':<8} {'±0.1정확도':<10} {'±0.2정확도':<10} {'±0.5정확도':<10}")
    print("-" * 100)
    
    for criterion, metrics in results['by_criterion'].items():
        print(f"{criterion:<12} {metrics['MSE']:<8.4f} {metrics['MAE']:<8.4f} {metrics['RMSE']:<8.4f} "
              f"{metrics['R2']:<8.4f} {metrics['Pearson']:<8.4f} {metrics['QWK']:<8.4f} "
              f"{metrics['Tolerance_0.1_Acc']:<10.4f} {metrics['Tolerance_0.2_Acc']:<10.4f} {metrics['Tolerance_0.5_Acc']:<10.4f}")
    
    if detailed:
        # 점수 분포 분석
        print("\n📈 점수 분포 분석:")
        print("-" * 80)
        print(f"{'평가기준':<12} {'실제평균':<10} {'예측평균':<10} {'실제표준편차':<12} {'예측표준편차':<12}")
        print("-" * 80)
        
        for criterion, dist in results['score_distribution'].items():
            print(f"{criterion:<12} {dist['true_mean']:<10.3f} {dist['pred_mean']:<10.3f} "
                  f"{dist['true_std']:<12.3f} {dist['pred_std']:<12.3f}")


def plot_evaluation_results(results, save_path=None):
    """평가 결과를 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 평가 기준별 MSE 비교
    criteria = list(results['by_criterion'].keys())
    mse_values = [results['by_criterion'][c]['MSE'] for c in criteria]
    
    axes[0, 0].bar(range(len(criteria)), mse_values, color='skyblue')
    axes[0, 0].set_title('평가 기준별 MSE')
    axes[0, 0].set_xlabel('평가 기준')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_xticks(range(len(criteria)))
    axes[0, 0].set_xticklabels(criteria, rotation=45, ha='right')
    
    # 2. 평가 기준별 Pearson 상관계수
    pearson_values = [results['by_criterion'][c]['Pearson'] for c in criteria]
    
    axes[0, 1].bar(range(len(criteria)), pearson_values, color='lightcoral')
    axes[0, 1].set_title('평가 기준별 Pearson 상관계수')
    axes[0, 1].set_xlabel('평가 기준')
    axes[0, 1].set_ylabel('Pearson 상관계수')
    axes[0, 1].set_xticks(range(len(criteria)))
    axes[0, 1].set_xticklabels(criteria, rotation=45, ha='right')
    axes[0, 1].set_ylim(0, 1)

    # 2. 평가 기준별 QWK
    pearson_values = [results['by_criterion'][c]['QWK'] for c in criteria]
    
    axes[0, 1].bar(range(len(criteria)), pearson_values, color='lightcoral')
    axes[0, 1].set_title('평가 기준별 QWK')
    axes[0, 1].set_xlabel('평가 기준')
    axes[0, 1].set_ylabel('QWK')
    axes[0, 1].set_xticks(range(len(criteria)))
    axes[0, 1].set_xticklabels(criteria, rotation=45, ha='right')
    axes[0, 1].set_ylim(0, 1)
    
    # 3. 실제 라벨 상관관계 히트맵
    im1 = axes[1, 0].imshow(results['correlation_matrix']['true_labels'], 
                           cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 0].set_title('실제 라벨 간 상관관계')
    axes[1, 0].set_xticks(range(len(criteria)))
    axes[1, 0].set_yticks(range(len(criteria)))
    axes[1, 0].set_xticklabels(criteria, rotation=45, ha='right')
    axes[1, 0].set_yticklabels(criteria)
    plt.colorbar(im1, ax=axes[1, 0])
    
    # 4. 예측 라벨 상관관계 히트맵
    im2 = axes[1, 1].imshow(results['correlation_matrix']['pred_labels'], 
                           cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('예측 라벨 간 상관관계')
    axes[1, 1].set_xticks(range(len(criteria)))
    axes[1, 1].set_yticks(range(len(criteria)))
    axes[1, 1].set_xticklabels(criteria, rotation=45, ha='right')
    axes[1, 1].set_yticklabels(criteria)
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_prediction_errors(true_labels, pred_labels, criterion_names=None, top_k=5):
    """예측 오차 상세 분석"""
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    if criterion_names is None:
        criterion_names = [f'기준_{i+1}' for i in range(11)]
    
    errors = np.abs(true_labels - pred_labels)
    
    print("\n🔍 예측 오차 상세 분석:")
    print("=" * 50)
    
    # 가장 큰 오차를 보이는 샘플들
    max_errors_per_sample = np.max(errors, axis=1)
    worst_samples = np.argsort(max_errors_per_sample)[-top_k:]
    
    print(f"\n가장 큰 오차를 보이는 상위 {top_k}개 샘플:")
    for i, sample_idx in enumerate(reversed(worst_samples)):
        max_error = max_errors_per_sample[sample_idx]
        worst_criterion_idx = np.argmax(errors[sample_idx])
        worst_criterion = criterion_names[worst_criterion_idx]
        
        print(f"  {i+1}. 샘플 {sample_idx}: 최대 오차 {max_error:.3f} ({worst_criterion})")
        print(f"     실제: {true_labels[sample_idx, worst_criterion_idx]:.3f}, "
              f"예측: {pred_labels[sample_idx, worst_criterion_idx]:.3f}")
    
    # 가장 어려운 평가 기준들
    mean_errors_per_criterion = np.mean(errors, axis=0)
    difficult_criteria = np.argsort(mean_errors_per_criterion)[-top_k:]
    
    print(f"\n가장 예측하기 어려운 상위 {top_k}개 평가 기준:")
    for i, criterion_idx in enumerate(reversed(difficult_criteria)):
        criterion = criterion_names[criterion_idx]
        mean_error = mean_errors_per_criterion[criterion_idx]
        print(f"  {i+1}. {criterion}: 평균 오차 {mean_error:.3f}")




