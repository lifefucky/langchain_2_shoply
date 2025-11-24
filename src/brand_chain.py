import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app_lc import Consultant, setup_api_config, FAQResponse
from langchain_openai import ChatOpenAI


class BrandStyleConsultant:
    def __init__(self):
        self.config = setup_api_config()
        self.consultant = Consultant(model_config=self.config)
        self.retrieval_chain = None
        self._init_components()

    def _init_components(self):
        """Инициализация LLM и RAG-цепочки"""
        # Корректная инициализация LLM с поддержкой кастомных эндпоинтов
        self.llm = ChatOpenAI(
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            model=self.config.llm_model,
            base_url=self.config.base_url or None
        )
        # Создание компонентов RAG через методы Consultant
        vector_store = self.consultant.create_vector_store()
        self.retrieval_chain = self.consultant.retrieval_chain(
            model=self.llm,
            vector_store=vector_store
        )

    def process_query(self, query: str) -> FAQResponse:
        """Обработка запроса с использованием структурированного вывода"""
        if query.startswith("/order"):
            response = self.consultant.orders_processor(query=query)
            self.consultant.add_to_history("assistant", response)
        else:
            self.consultant.add_to_history("user", query)
            response = self.consultant.faq_processor(
                query=query,
                retrieval_chain=self.retrieval_chain
            )
            self.consultant.add_to_history("assistant", response.answer)
        return response