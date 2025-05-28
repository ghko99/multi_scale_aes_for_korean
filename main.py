import pandas as pd
import torch
from model_architecture_bert_multi_scale_multi_loss import DocumentBertScoringModel

def load_data():
    train = pd.read_csv('./data/train.csv',encoding='utf-8-sig')
    test = pd.read_csv('./data/valid.csv',encoding='utf-8-sig')

    train_essays = train['essay'].to_list()
    train_score_avg = train['essay_score_avg'].to_list()

    test_essays = test['essay'].to_list()
    test_score_avg = test['essay_score_avg'].to_list()

    train_labels = []
    test_labels = []

    for i in range(len(train_score_avg)):
        train_scores = train_score_avg[i].split('#')
        train_scores = [float(score) for score in train_scores]
        train_labels.append(train_scores)

    for i in range(len(test_score_avg)):
        test_scores = test_score_avg[i].split('#')
        test_scores = [float(score) for score in test_scores]
        test_labels.append(test_scores)
    
    return (train_essays, train_labels), (test_essays, test_labels)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 디바이스: {device}")
    CRITERION_NAMES = [
                '문법 정확도', '단어 선택의 적절성', '문장 표현', '문단 내 구조', '문단 간 구조',
                '구조의 일관성', '분량의 적절성', '주제 명료성', '창의성', 
                '프롬프트 독해력', '설명의 구체성'
    ]
    (train_essays, train_labels), (test_essays,test_labels) = load_data()

    model = DocumentBertScoringModel(load_model=False)

    df_train_essays = pd.DataFrame({'essay': train_essays})
    df_train_labels = pd.DataFrame(train_labels, columns=CRITERION_NAMES)

    df_test_essays = pd.DataFrame({'essay': test_essays})
    df_test_labels = pd.DataFrame(test_labels, columns=CRITERION_NAMES)

    train_data = (df_train_essays['essay'], df_train_labels)
    test_data = (df_test_essays['essay'], df_test_essays)

    model.fit(train_data, test=test_data, mode='2025_05_29')