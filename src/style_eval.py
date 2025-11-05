import os, json, pathlib, re, statistics
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from brand_chain import ask, STYLE, BASE

load_dotenv(BASE / ".env", override=True)
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)

# Простые проверки до LLM
def rule_checks(text: str) -> int:
    score = 100
    # 1) Без эмодзи
    if re.search(r"[\U0001F300-\U0001FAFF]", text):
        score -= 20
    # 2) Без крика!!!
    if "!!!" in text:
        score -= 10
    # 3) Длина
    if len(text) > 600:
        score -= 10
    return max(score, 0)

# LLM-оценка
class Grade(BaseModel):
    score: int = Field(..., ge=0, le=100)
    notes: str

LLM = ChatOpenAI(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"), temperature=0)

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"Ты — строгий ревьюер соответствия голосу бренда {STYLE['brand']}"),
    ("system", f"Тон: {STYLE['tone']['persona']}. Избегай: {', '.join(STYLE['tone']['avoid'])}. "
               f"Обязательно: {', '.join(STYLE['tone']['must_include'])}."),
    ("human", "Ответ ассистента:\n{answer}\n\nДай целочисленный score 0..100 и краткие заметки почему.")
])

def llm_grade(text: str) -> Grade:
    parser = LLM.with_structured_output(Grade)
    return (GRADE_PROMPT | parser).invoke({"answer": text})

def eval_batch(prompts: List[str]) -> dict:
    results = []
    for p in prompts:
        reply = ask(p)
        rule = rule_checks(reply.answer)
        g = llm_grade(reply.answer)
        final = int(0.4 * rule + 0.6 * g.score)
        results.append({
            "prompt": p,
            "answer": reply.answer,
            "actions": reply.actions,
            "tone_model": reply.tone,
            "rule_score": rule,
            "llm_score": g.score,
            "final": final,
            "notes": g.notes
        })
    mean_final = round(statistics.mean(r["final"] for r in results), 2)
    out = {"mean_final": mean_final, "items": results}
    (REPORTS / "style_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

if __name__ == "__main__":
    eval_prompts = (BASE / "data/eval_prompts.txt").read_text(encoding="utf-8").strip().splitlines()
    report = eval_batch(eval_prompts)
    print("Средний балл:", report["mean_final"])
    print("Отчёт:", REPORTS / "style_eval.json")
