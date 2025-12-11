#!/usr/bin/env python3
"""
Router 학습용 데이터 생성 스크립트 (v6)

도메인별 구조가 다른 법률 데이터(JSON 파일들)에서 'input'(질문문)을 추출하여
Router 학습용 JSONL 파일로 저장합니다.

- 민사법(civil), 지식재산법(ip): taskinfo 계열
- 형사법(criminal), 행정법(admin): label 계열
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

# ======================
# 설정
# ======================
DATA_ROOT = Path("/workspace/data")
OUTPUT_FILE = Path("/workspace/router_train/domain_router_train.jsonl")
MAX_SAMPLES_PER_DOMAIN = 2000

# 도메인 경로 매핑
DOMAIN_CONFIG = {
    "01.민사법 LLM 사전학습 및 Instruction Tuning 데이터": "civil",
    "04.형사법 LLM 사전학습 및 Instruction Tuning 데이터": "criminal",
    "02.지식재산권법 LLM 사전학습 및 Instruction Tuning 데이터": "ip",
    "03.행정법 LLM 사전학습 및 Instruction Tuning 데이터": "admin"
}


# ======================
# 헬퍼 함수
# ======================
def find_labeling_folder(training_dir: Path):
    """Training 폴더 내에서 '라벨링' 포함 폴더 찾기"""
    for item in training_dir.iterdir():
        if item.is_dir() and "라벨링" in item.name:
            return item
    return None


def collect_json_files(labeling_dir: Path):
    """라벨링 폴더 내 모든 JSON 파일 수집"""
    json_files = []
    for root, _, files in os.walk(labeling_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(Path(root) / file)
    return json_files


def extract_input_text(data: dict):
    """JSON 구조에서 '질문문(input)' 추출"""

    # taskinfo 계열 (민사, 지식재산법)
    if "taskinfo" in data and isinstance(data["taskinfo"], dict):
        t = data["taskinfo"]
        if "input" in t and isinstance(t["input"], str):
            return t["input"].strip()

    # label 계열 (형사, 행정법)
    if "label" in data and isinstance(data["label"], dict):
        t = data["label"]
        if "input" in t and isinstance(t["input"], str):
            return t["input"].strip()

    # 혹시 모를 fallback
    for key in ["input", "question", "질문"]:
        if key in data and isinstance(data[key], str):
            return data[key].strip()

    return None


# ======================
# 메인 처리 함수
# ======================
def process_domain(domain_folder: str, label: str, max_samples: int):
    """도메인별 데이터 처리"""
    samples = []
    base_dir = DATA_ROOT / domain_folder / "3.개방데이터" / "1.데이터" / "Training"
    if not base_dir.exists():
        print(f"⚠️ 경로 없음: {base_dir}")
        return samples

    labeling_dir = find_labeling_folder(base_dir)
    if not labeling_dir:
        print(f"⚠️ 라벨링 폴더 없음: {base_dir}")
        return samples

    print(f"📂 {label.upper()} 도메인 처리 중: {labeling_dir}")
    json_files = collect_json_files(labeling_dir)
    print(f"   총 {len(json_files)}개 JSON 파일 발견")

    for file in tqdm(json_files, desc=f"   {label}", leave=False):
        if len(samples) >= max_samples:
            break
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = extract_input_text(data)
            if text and len(text) > 5:
                samples.append({"text": text, "label": label})
        except Exception:
            continue

    print(f"   ✅ {len(samples)}개 샘플 수집 완료\n")
    return samples


# ======================
# 메인 실행
# ======================
def main():
    print("=" * 60)
    print("🚀 Router 학습 데이터 생성 (v6)")
    print("=" * 60)
    print()

    all_samples = []

    for folder, label in DOMAIN_CONFIG.items():
        samples = process_domain(folder, label, MAX_SAMPLES_PER_DOMAIN)
        all_samples.extend(samples)

    # 중복 제거
    unique_texts = {}
    filtered_samples = []
    for s in all_samples:
        if s["text"] not in unique_texts:
            unique_texts[s["text"]] = True
            filtered_samples.append(s)
    all_samples = filtered_samples

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("📊 데이터 수집 결과")
    print("=" * 60)
    counts = {}
    for s in all_samples:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    for k, v in counts.items():
        print(f"{k:10s}: {v}개")
    print(f"총합: {len(all_samples)}개")
    print(f"💾 저장 완료: {OUTPUT_FILE}")
    print("=" * 60)
    print("\n샘플 미리보기:")
    for i, s in enumerate(all_samples[:3], 1):
        print(f"[{i}] ({s['label']}) {s['text']}")
    print()


if __name__ == "__main__":
    main()
