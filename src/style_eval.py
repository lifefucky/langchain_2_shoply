import os, json, pathlib, re, statistics
from datetime import datetime

from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate
import yaml

from brand_chain import BrandStyleConsultant
from styles_prompt import StyleParser
from logging_config import setup_logger


BASE = pathlib.Path(__file__).parent.parent

# LLM-оценка
class Grade(BaseModel):
    score: int = Field(..., ge=0, le=100)
    notes: str

class ReportMaker:
    def __init__(self, base_path: pathlib.Path):
        self.base_path = base_path
        self.report_path = pathlib.Path(self.base_path).joinpath('reports')
        os.makedirs(self.report_path, exist_ok=True)
        self._model_config = None
        self._test_bot = None

        now: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logger(name=__name__, log_file=f"session_{now}.jsonl")
        self.add_log(event='initialized')

    @property
    def model_config(self):
        if self._model_config is None:
            with open(pathlib.Path(self.base_path).joinpath('model_configs.yaml'), mode='r', encoding='utf-8') as f:
                configs = yaml.safe_load(f)['prompts']['report_answer']

            current_version = configs['current_version']
            self._model_config = configs['versions'][current_version]
            self.add_log(event='model_config_loaded', version=current_version)
        return self._model_config

    @property
    def test_bot(self):
        if self._test_bot is None:
            self._test_bot = BrandStyleConsultant()
            self.add_log(event='test_bot_initialized')
        return self._test_bot

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

    def llm_config(self):
        load_dotenv(self.base_path / ".env", override=True)

        # Проверяем переменные окружения
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_API_BASE_URL")
        llm_model = os.getenv("LLM_MODEL")
        temperature = float(os.getenv("LLM_TEMPERATURE"))

        if _temperature := self.model_config.get('temperature'):
            temperature = float(_temperature)

        if _llm_model := self.model_config.get('model'):
            llm_model = _llm_model

        return {
            "api_key": api_key,
            "temperature": temperature,
            "model_name": llm_model,
            "openai_api_base": base_url}

    @property
    def style(self):
        if not hasattr(self, '_style'):
            self._style = StyleParser().style
        return self._style

    def _system_prompt(self, template):
        prompt_template = PromptTemplate(
            template=template,
            input_variables=['brand_name', 'tone', 'avoid', 'must_include']
        )
        return SystemMessagePromptTemplate(
            prompt=prompt_template.partial(
                brand_name=self.style.brand,
                tone=self.style.persona,
                avoid=', '.join(self.style.avoid),
                must_include=', '.join(self.style.must_include)
            )
        )

    def llm_grade(self, text):
        try:
            LLM = ChatOpenAI(**self.llm_config())

            if system_template := self.model_config['prompt']['system']:
                system_message = self._system_prompt(system_template)

            human_prompt = HumanMessagePromptTemplate.from_template(self.model_config['prompt']['user'])
            human_message = human_prompt.format(answer=text)

            GRADE_PROMPT = ChatPromptTemplate.from_messages([
                system_message, human_message
            ])

            parser = LLM.with_structured_output(Grade)
            result = (GRADE_PROMPT | parser).invoke({"answer": text})
            self.add_log(event='llm_grade_success', score=result.score)
        except Exception as e:
            self.add_log(type='error', event='llm_grade_failed', error=str(e))
            return Grade(score=50, notes=f"Ошибка оценки: {str(e)}")
        return result


    def eval_batch(self, prompts: List[str]) -> dict:
        self.add_log(event='eval_batch_started', total_prompts=len(prompts))
        test_bot = self.test_bot

        results = []
        for idx, p in enumerate(prompts):
            try:
                reply = test_bot.process_query(query=p)

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

                self.add_log(
                    event='prompt_evaluated',
                    index=idx,
                    final_score=final
                    )
            except Exception as e:
                self.add_log(
                    type='error',
                    event='evaluation_failed',
                    prompt_index=idx,
                    error=str(e)
                )
                raise

        mean_final = round(statistics.mean(r["final"] for r in results), 2)
        out = {"mean_final": mean_final, "items": results}
        report_file = self.report_path / "style_eval.json"
        report_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        self.add_log(
            event='eval_batch_completed',
            total_prompts=len(prompts),
            mean_score=mean_final,
            report_file=str(report_file)
        )
        return out

    def add_log(self, type: str = "info", query: str = None, **kwargs):
        log_entry = {
            "component": "ReportMaker",
            "query": query,
            **kwargs
        }
        if type == 'error':
            self.logger.error(json.dumps(log_entry, ensure_ascii=False))
        else:
            self.logger.info(json.dumps(log_entry, ensure_ascii=False))

if __name__ == "__main__":
    r_maker = ReportMaker(base_path=BASE)

    eval_prompts = (BASE / "data/eval_prompts.txt").read_text(encoding="utf-8").strip().splitlines()
    report = r_maker.eval_batch(eval_prompts)
    print("Средний балл:", report["mean_final"])
    print("Отчёт:", r_maker.report_path / "style_eval.json")
