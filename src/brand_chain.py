import os
from dataclasses import dataclass

import yaml
from pathlib import Path

from langchain_openai import ChatOpenAI

from app_lc import setup_api_config, Consultant, ModelConfig

'''
+Нужно, чтобы модель отвечала в формате answer, tone, actions
app_lc: Доработать промпт, чтобы если статус заказа - предложить выполнить команду
app_lc: System и User prompt 
app_lc: Добавить few_shots и использование в промпте
app_lc: tone of voice добавить
style_eval: собрать метод для обработки из style_eval.py

доработать style_eval.py
доработать brand_chain.py
+Ответы парсятся в Pydantic-модель (структурированный вывод) или через with_structured_output.
+Доработать FAQ - если вопрос касается статуса заказа - предложить выполнить команду
'''


@dataclass
class StyleGuide:
    brand_name: str
    tone: str
    avoid: str
    must_include: str


class BrandChain:
    def __init__(self):
        self.base_path = Path('..')

        with open(Path(self.base_path).joinpath('data/style_guide.yaml'), mode='r', encoding='utf-8') as f:
            self.style_guide = yaml.safe_load(f)

    def base(self):
        print(self.base_path)
        return self.base_path

    def style(self) -> StyleGuide:
        style_guide = self.style_guide
        return StyleGuide(brand_name=style_guide['brand'],
                          tone=style_guide['tone']['persona'],
                          avoid=style_guide['tone']['avoid'],
                          must_include=', '.join(style_guide['tone']['must_include']))


class TestRun:
    def __init__(self):
        self.bot = self.build_bot(setup_api_config())

    def build_bot(self, model_config: ModelConfig):
        return Consultant(model_config=model_config)

    def retrieval_chain(self):
        config = self.bot.model_config
        self.bot.add_log(event="config_loaded", config=config.to_dict())

        llm_kwargs = {
            "api_key": config.api_key,
            "temperature": config.temperature,
            "model_name": config.llm_model,
            "openai_api_base": config.base_url}
        model = ChatOpenAI(**llm_kwargs)
        vector_store = self.bot.create_vector_store()
        return self.bot.retrieval_chain(model=model, vector_store=vector_store)

    def process_query(self, retrieval_chain, query: str):

        if query.startswith("/order"):
            response = self.bot.orders_processor(query=query)
            self.bot.add_to_history("assistant", response)
        else:
            response = self.bot.faq_processor(query=query, retrieval_chain=retrieval_chain)
            self.bot.add_to_history("assistant", response)

        return response
