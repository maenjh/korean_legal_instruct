import json
import random

# 각 도메인별 질문 템플릿 및 키워드
domain_templates = {
    "civil": {
        "keywords": [
            "계약", "손해배상", "불법행위", "소유권", "임대차", "매매", "채권", "채무", 
            "부동산", "상속", "이혼", "양육권", "위자료", "명예훼손", "프라이버시", 
            "소비자", "제조물책임", "의료과실", "교통사고", "보험금", "근로계약",
            "부당해고", "임금", "퇴직금", "산재보상", "민사소송", "가압류", "경매",
            "파산", "회생", "담보권", "저당권", "전세권", "지상권", "용익권"
        ],
        "templates": [
            "{}에 대한 법적 책임은 어떻게 되나요?",
            "{}와 관련된 소송을 제기하려면 어떤 절차가 필요한가요?",
            "{}의 법적 효력은 무엇인가요?",
            "{}에 대한 손해배상 청구는 어떻게 하나요?",
            "{}의 성립 요건은 무엇인가요?",
            "{}가 인정되기 위한 법적 기준은 무엇인가요?",
            "{}에 관한 분쟁이 발생했을 때 어떻게 해결하나요?",
            "{}의 시효는 얼마나 되나요?",
            "{}를 증명하기 위해 필요한 증거는 무엇인가요?",
            "{}의 취소나 해제는 어떤 경우에 가능한가요?"
        ]
    },
    "criminal": {
        "keywords": [
            "절도", "사기", "횡령", "배임", "폭행", "상해", "살인", "강도", "강간",
            "성폭력", "아동학대", "가정폭력", "명예훼손", "모욕", "협박", "공갈",
            "마약", "뇌물", "위증", "무고", "업무방해", "방화", "문서위조", "인장위조",
            "공무집행방해", "도주", "음주운전", "무면허운전", "교통사고", "과실치사상",
            "체포", "구속", "수사", "기소", "형량", "집행유예", "선고유예", "보석"
        ],
        "templates": [
            "{}죄로 기소된 경우 예상되는 형량은 어떻게 되나요?",
            "{}의 성립 요건은 무엇인가요?",
            "{}와 {}의 차이점은 무엇인가요?",
            "{}혐의로 조사를 받을 때 주의할 점은 무엇인가요?",
            "{}사건에서 피해자가 취할 수 있는 법적 조치는 무엇인가요?",
            "{}로 고소를 당했을 때 어떻게 대응해야 하나요?",
            "{}의 공소시효는 얼마나 되나요?",
            "{}에 대한 처벌 규정은 어떻게 되나요?",
            "{}미수는 처벌되나요?",
            "{}의 정당방위가 인정되는 경우는 언제인가요?"
        ]
    },
    "admin": {
        "keywords": [
            "행정처분", "영업정지", "과태료", "과징금", "인허가", "등록", "신고",
            "행정심판", "행정소송", "취소소송", "무효소송", "부작위위법확인소송",
            "공무원", "징계", "면직", "해임", "국가배상", "손실보상", "공용수용",
            "도시계획", "건축허가", "개발행위허가", "환경영향평가", "정보공개",
            "개인정보보호", "조세", "부가가치세", "종합소득세", "법인세", "상속세",
            "증여세", "취득세", "재산세", "체납", "압류", "공매"
        ],
        "templates": [
            "{}에 대한 이의신청 절차는 어떻게 되나요?",
            "{}가 부당하다고 생각될 때 취할 수 있는 법적 조치는 무엇인가요?",
            "{}를 받기 위한 요건은 무엇인가요?",
            "{}의 법적 근거는 무엇인가요?",
            "{}에 대한 취소를 구하려면 어떻게 해야 하나요?",
            "{}의 부과 기준은 무엇인가요?",
            "{}와 관련된 행정심판을 제기할 수 있는 기간은 얼마나 되나요?",
            "{}가 위법한지 판단하는 기준은 무엇인가요?",
            "{}에 불복하려면 어떤 절차를 거쳐야 하나요?",
            "{}의 효력은 언제 발생하나요?"
        ]
    },
    "ip": {
        "keywords": [
            "특허", "실용신안", "디자인", "상표", "저작권", "영업비밀", "부정경쟁방지",
            "특허출원", "특허등록", "특허무효", "특허침해", "상표등록", "상표침해",
            "저작권침해", "표절", "복제권", "배포권", "공연권", "전송권", "2차적저작물",
            "공동저작물", "업무상저작물", "저작인격권", "저작재산권", "라이선스", "기술이전",
            "직무발명", "발명자", "권리범위", "균등론", "선사용권", "실시권", "통상실시권",
            "전용실시권", "침해금지", "손해배상", "부당이득반환"
        ],
        "templates": [
            "{}의 보호 요건은 무엇인가요?",
            "{}가 인정되려면 어떤 조건을 충족해야 하나요?",
            "{}침해로 인한 손해배상은 어떻게 청구하나요?",
            "{}와 {}의 차이점은 무엇인가요?",
            "{}의 등록 절차는 어떻게 되나요?",
            "{}의 보호기간은 얼마나 되나요?",
            "{}가 무효가 되는 경우는 언제인가요?",
            "{}분쟁에서 승소하기 위해 필요한 증거는 무엇인가요?",
            "{}의 권리범위는 어떻게 해석하나요?",
            "{}계약 시 주의해야 할 사항은 무엇인가요?"
        ]
    }
}

def generate_questions(domain, count=2000):
    """특정 도메인에 대한 질문 생성"""
    questions = []
    keywords = domain_templates[domain]["keywords"]
    templates = domain_templates[domain]["templates"]
    
    # 기본 질문 생성
    for i in range(count):
        keyword = random.choice(keywords)
        template = random.choice(templates)
        
        # 일부 템플릿은 두 개의 키워드를 사용
        if "{}" in template and template.count("{}") == 2:
            keyword2 = random.choice([k for k in keywords if k != keyword])
            question = template.format(keyword, keyword2)
        else:
            question = template.format(keyword)
        
        questions.append({
            "text": question,
            "label": domain
        })
    
    return questions

def main():
    """모든 도메인에 대한 질문 생성 및 저장"""
    all_questions = []
    
    print("도메인별 질문 생성 중...")
    for domain in ["civil", "criminal", "admin", "ip"]:
        print(f"  - {domain}: 2000개 생성 중...")
        questions = generate_questions(domain, 2000)
        all_questions.extend(questions)
        print(f"    완료! (총 {len(questions)}개)")
    
    # 질문 섞기
    random.shuffle(all_questions)
    
    # JSONL 파일로 저장
    output_file = "/workspace/router_train/domain_router_train.jsonl"
    print(f"\n저장 중: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_questions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n완료! 총 {len(all_questions)}개의 질문이 생성되었습니다.")
    
    # 도메인별 개수 확인
    print("\n도메인별 개수:")
    domain_counts = {}
    for item in all_questions:
        label = item['label']
        domain_counts[label] = domain_counts.get(label, 0) + 1
    
    for domain, count in sorted(domain_counts.items()):
        print(f"  - {domain}: {count}개")

if __name__ == "__main__":
    main()
