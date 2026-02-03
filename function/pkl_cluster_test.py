import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import cv2
import sys

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def clean_failure(x):
    if isinstance(x, list) and len(x) > 0:
        val = x[0]
        if isinstance(val, (list, tuple)): return str(val[0])
        return str(val)
    return str(x)

def check_cluster_mapping(pkl_path):
    print(f"=== 클러스터 매핑 분석 시작: {os.path.basename(pkl_path)} ===")
    
    # 1. Load Data
    try:
        df = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return

    if isinstance(df, list): df = pd.DataFrame(df)
    
    if 'failure_type' in df.columns:
        df['failure_type'] = df['failure_type'].apply(clean_failure)
    else:
        print("'failure_type' 컬럼을 찾을 수 없습니다.")
        return

    # Filter targets
    TARGET_LABELS = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None']
    df = df[df['failure_type'].isin(TARGET_LABELS)].reset_index(drop=True)
    
    print(f"총 칩 개수: {len(df)}")
    
    # 2. Load Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    try:
        # ResNet
        model_path = os.path.join(MODELS_DIR, 'wafer_classifier.pth')
        base_model = models.resnet18(weights=None)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base_model.fc = nn.Linear(base_model.fc.in_features, 9)
        state = torch.load(model_path, map_location=device)
        if 'state_dict' in state: state = state['state_dict']
        new_state = {}
        for k, v in state.items():
            if k.startswith('model.'): new_state[k[6:]] = v
            else: new_state[k] = v
        base_model.load_state_dict(new_state, strict=False)
        resnet = nn.Sequential(*list(base_model.children())[:-1]).to(device).eval()
        
        # KMeans
        kmeans_path = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
        kmeans = joblib.load(kmeans_path)
        
        print("AI 모델 로드 완료.")
        
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return

    # 3. Features & Clustering
    print("특징 벡터 추출 중...")
    features_list = []
    
    with torch.no_grad():
        for i, row in df.iterrows():
            if i % 100 == 0: print(f"진행 중 {i}/{len(df)}...", end='\r')
            tsv = row.get('tsv_matrix')
            if not isinstance(tsv, np.ndarray): tsv = np.array(tsv)
            
            img = cv2.resize(tsv.astype('float32'), (64, 64), interpolation=cv2.INTER_NEAREST)
            img = img / 2.0
            tensor_img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            feat = resnet(tensor_img)
            feat = feat.view(feat.size(0), -1).cpu().numpy()
            features_list.append(feat)
            
    print("\n클러스터링 수행 중...")
    X_features = np.concatenate(features_list, axis=0)
    cluster_labels = kmeans.predict(X_features)
    df['cluster_label'] = cluster_labels
    
    # 4. Analyze Mapping
    print("\n=== 클러스터 ID -> 불량 타입 매핑 분석 결과 ===")
    unique_clusters = np.sort(np.unique(cluster_labels))
    
    mapping_suggestion = {}
    
    for c_id in unique_clusters:
        subset = df[df['cluster_label'] == c_id]
        counts = subset['failure_type'].value_counts()
        total_in_cluster = len(subset)
        
        top_failure = counts.index[0] if not counts.empty else "Unknown"
        top_count = counts.iloc[0] if not counts.empty else 0
        ratio = (top_count / total_in_cluster * 100) if total_in_cluster > 0 else 0
        
        print(f"\n클러스터 {c_id} (총 {total_in_cluster}개):")
        if not counts.empty:
            for f_type, count in counts.head(3).items():
                print(f"  - {f_type}: {count} ({count/total_in_cluster*100:.1f}%)")
        
        print(f"  -> 대표 불량 타입: {top_failure} ({ratio:.1f}%)")
        mapping_suggestion[c_id] = top_failure
        
    print("\n=== 추천 매핑 코드 (복사해서 사용하세요) ===")
    print("CLUSTER_TO_PATTERN = {")
    for k, v in mapping_suggestion.items():
        print(f"    {k}: \"{v}\",")
    print("}")

if __name__ == "__main__":
    # 타겟 PKL 파일 경로 (직접 수정해서 사용)
    # 가장 최신 파일 자동 탐색
    files = [f for f in os.listdir(MODELS_DIR) if f.startswith('BATCH_') and f.endswith('.pkl')]
    if files:
        files.sort(key=lambda x: x[6:21] if len(x) >= 21 else "", reverse=True)
        target_pkl = os.path.join(MODELS_DIR, files[0])
        check_cluster_mapping(target_pkl)
    else:
        print("models 디렉토리에서 배치 파일을 찾을 수 없습니다.")
