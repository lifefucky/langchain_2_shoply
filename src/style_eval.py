import os, json, pathlib, re, statistics
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import yaml

from app_lc import ModelConfig
from brand_chain import TestRun, BrandChain




'''
Берутся промпты из eval_prompts.txt
Прогоняются ответы
Считается статистика
Вывод

TODO:
оптимизировать ввод промпта проверки


'''

@dataclass
class PromptDetails:
    prompt: str
    temperature: float
    model: str = ""

# LLM-оценка
class Grade(BaseModel):
    score: int = Field(..., ge=0, le=100)
    notes: str

class ReportMaker:
    def __init__(self, base_path: pathlib.Path, test_bot: TestRun):
        self.base_path = base_path

        os.makedirs(pathlib.Path(self.base_path).joinpath('reports'), exist_ok=True)
        self.model_config = self.model_config()
        self.test_bot = test_bot

    @staticmethod
    def rule_checks(text: str) -> int:
        # Простые проверки до LLM
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

    def model_config(self):
        with open(pathlib.Path(self.base_path).joinpath('model_configs.yaml'), mode='r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)
        details = configs['prompts']['report_answer']['versions'][ configs['prompts']['report_answer']['current_version'] ]
        return details

    def llm_config(self):
        load_dotenv(self.base_path / ".env", override=True)

        # Проверяем переменные окружения
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_API_BASE_URL")
        llm_model = os.getenv("LLM_MODEL")
        temperature = float(os.getenv("LLM_TEMPERATURE"))

        if _temperature := self.model_config['temperature']:
            temperature = float(_temperature)

        if _llm_model := self.model_config['model']:
            llm_model = _llm_model

        return {
            "api_key": api_key,
            "temperature": temperature,
            "model_name": llm_model,
            "openai_api_base": base_url}

    def llm_grade(self, text):
        LLM = ChatOpenAI(**self.llm_config())

        system_prompt = SystemMessagePromptTemplate.from_template(self.model_config['prompt']['system'])
        system_message = system_prompt.format(brand_name=STYLE['brand'],
                                              tone=STYLE['tone']['persona'],
                                              avoid=STYLE['tone']['avoid'],
                                              must_include=', '.join(STYLE['tone']['must_include']))

        human_prompt = HumanMessagePromptTemplate.from_template(self.model_config['prompt']['user'])
        human_message = human_prompt.format(answer=text)

        GRADE_PROMPT = ChatPromptTemplate.from_messages([
            system_message, human_message
        ])

        parser = LLM.with_structured_output(Grade)
        return (GRADE_PROMPT | parser).invoke({"answer": text})


    def eval_batch(self, prompts: List[str]) -> dict:
        retrieval_chain = self.test_bot.retrieval_chain()

        results = []
        for p in prompts:
            reply = self.test_bot.process_query(retrieval_chain=retrieval_chain, query=p)

            rule = self.rule_checks(reply.answer)
            g = self.llm_grade(reply.answer)
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
    brand_chain = BrandChain()

    BASE = brand_chain.base()
    STYLE = brand_chain.style()

    test_run = TestRun()

    r_maker = ReportMaker(base_path=BASE, test_bot=test_run)

    eval_prompts = (BASE / "data/eval_prompts.txt").read_text(encoding="utf-8").strip().splitlines()
    report = eval_batch(eval_prompts)
    print("Средний балл:", report["mean_final"])
    print("Отчёт:", REPORTS / "style_eval.json")
