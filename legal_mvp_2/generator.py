from router import DomainRouter
from llm_manager import LLMManager
import config

class ResponseGenerator:
    def __init__(self):
        self.router = DomainRouter()
        self.llm_manager = LLMManager()
        
        # Korean mapping for display/prompt
        self.domain_names_kr = {
            "civil": "민사법",
            "criminal": "형사법",
            "admin": "행정법",
            "ip": "지식재산권법"
        }

    def _format_history(self, history: list) -> str:
        if not history:
            return "없음"
        
        formatted = []
        for turn in history:
            role = "User" if turn.get("role") == "user" else "AI"
            content = turn.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def _construct_prompt(self, domain_key: str, user_input: str, history: list) -> str:
        domain_kr = self.domain_names_kr.get(domain_key, "법률")
        history_text = self._format_history(history)
        
        # Template from requirements
        prompt = f"""### 역할:
당신은 {domain_kr} 분야의 친절하고 전문적인 AI 법률 상담사입니다.
사용자는 법적 조언을 구하는 일반인('의뢰인')입니다. 판사가 아닙니다.

### 🚨 절대 금지 사항 (Strict Constraints):
1. **사용자를 '피고인', '피의자'라고 부르지 마십시오.** 반드시 '의뢰인' 또는 '귀하'라고 부르십시오.
2. **과거의 판결문처럼 말하지 마십시오.** (예: "법원은 ~라고 판단하였다", "피고인은 ~하였다" 금지)
3. **확정적인 유/무죄 판결을 내리지 마십시오.** 가능성과 조건을 설명하십시오.
4. **없는 사실을 지어내지 마십시오.** 사용자가 말한 내용만으로 판단하십시오.

### 답변 작성 가이드:
1. **공감과 경청**: 의뢰인의 상황에 공감하며 부드러운 대화체(해요체)로 시작하십시오.
2. **법적 쟁점 설명**: 의뢰인의 상황에서 어떤 법적 요건이 중요한지 설명하십시오.
3. **조건부 예측**: "만약 ~라면 ~될 수 있습니다" 형태로 다양한 가능성을 열어두고 설명하십시오.
4. **Markdown 활용**: 가독성을 위해 헤더(###)와 볼드체(**)를 적절히 사용하십시오.

### 답변 구조:
### 1. 상황 공감 및 정리
(의뢰인의 상황을 듣고 공감하며 요약)

### 2. 주요 법적 쟁점
(해당 사건에서 문제가 되는 법적 포인트 설명)

### 3. 예상되는 결과 및 대응 방안
(유리한 점과 불리한 점, 앞으로 준비해야 할 것들)

### 질문:
{user_input}

### 이전 대화:
{history_text}

### 답변:
"""
        return prompt

    def generate(self, message: str, history: list, manual_domain: str = None):
        # 1. Determine Domain
        if manual_domain:
            domain_key = manual_domain
            print(f"[Generator] Manual domain selected: {domain_key}")
        else:
            # Use router
            # Combine history and current message for better context, or just use message?
            # app.py used all user turns. Let's do that.
            user_turns = [turn["content"] for turn in history if turn["role"] == "user"]
            user_turns.append(message)
            router_input = "\n".join(user_turns)
            
            domain_key = self.router.predict(router_input)
            print(f"[Generator] Auto-detected domain: {domain_key}")

        # 2. Set Adapter
        self.llm_manager.set_adapter(domain_key)

        # 3. Construct Prompt
        prompt = self._construct_prompt(domain_key, message, history)
        print(f"--- [Prompt] ---\n{prompt}\n----------------")

        # 4. Generate Response
        response_text = self.llm_manager.generate_response(prompt)
        
        return {
            "predicted_domain": domain_key,
            "answer": response_text,
            "history": history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response_text}
            ]
        }
