import argparse
import json

import logging
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from dataclasses import dataclass
from pydantic import BaseModel, Field, conlist
import yaml


#actions пустой

class FAQResponse(BaseModel):
    answer: str = Field(..., description="Краткий ответ на вопрос пользователя")
    tone: str = Field(..., description="Контроль тона: 'да' если соответствует бренду, иначе 'нет' с кратким объяснением (например: 'нет, слишком формальный')")
    actions: conlist(str, max_length=3) = Field(
        default_factory=list,
        description="Список из 1-3 следующих шагов для клиента"
    )

@dataclass
class ModelConfig:
    """Конфигурация для LLM и embeddings"""
    api_key: str
    base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.3
    context_length: int = 3
    brand_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует конфигурацию в словарь с маскировкой API ключа.
        Возвращает:
            Словарь с безопасными данными конфигурации
        """
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "context_length": self.context_length
        }


def setup_api_config() -> ModelConfig:
    """Настройка API ключа с приоритетом: переменные окружения -> .env файл -> ручной ввод"""
    # Загружаем .env файл, если он существует
    load_dotenv()

    # Проверяем переменные окружения
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_API_BASE_URL")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    llm_model = os.getenv("LLM_MODEL")
    temperature = float(os.getenv("LLM_TEMPERATURE"))
    context_length = int(os.getenv("CONTEXT_LENGTH"))
    brand_name = os.getenv("BRAND_NAME")

    return ModelConfig(
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model,
        llm_model=llm_model,
        temperature=temperature,
        context_length=context_length,
        brand_name=brand_name
    )


class Consultant:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(script_dir, "data")
        with open(os.path.join(self.data_dir, "faq.json"), mode='r', encoding='utf-8') as file:
            self.faq = json.load(file)
        with open(os.path.join(self.data_dir, "orders.json"), mode='r', encoding="utf-8") as file:
            self.orders = json.load(file)
        with open(os.path.join(script_dir, 'model_configs.yaml'), mode='r', encoding='utf-8') as file:
            self._prompt_config = yaml.safe_load(file)

        os.makedirs("logs", exist_ok=True)
        now: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = self.setup_logger(f"logs/session_{now}.jsonl")

        self.conversation_history: list[dict[str, str]] = []
        self.context_length = self.model_config.context_length

        self.few_shot_examples = self._load_few_shot_examples()
        self.example_selector = self._init_example_selector()

    @staticmethod
    def setup_logger(log_file):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        return logger

    def format_conversation_history(self) -> str:
        """Форматирует историю диалога в строку"""
        return "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}"
             for msg in self.conversation_history[-self.context_length:]]
        )

    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def prepare_text_faq(self) -> List[str]:
        return [f"Вопрос:'{qa['q']}'\nОтвет:'{qa['a']}'" for qa in self.faq]

    def create_vector_store(self) -> FAISS:
        texts = self.prepare_text_faq()

        try:
            # Настройка embedding модели
            embedding_kwargs = {
                "api_key": self.model_config.api_key,
                "model": self.model_config.embedding_model
            }

            if self.model_config.base_url:
                embedding_kwargs["base_url"] = self.model_config.base_url

            embeddings = OpenAIEmbeddings(**embedding_kwargs)

            documents = [Document(page_content=text) for text in texts]

            text_splitter = CharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separator="\n"
            )
            docs = text_splitter.split_documents(documents)

            vector_store = FAISS.from_documents(docs, embeddings)
            self.add_log(event='vector_store_created', document_count=len(docs), chunks_count=len(docs))
            return vector_store

        except Exception as e:
            details = {
                "documents_count": len(texts) if 'texts' in locals() else 0
            }
            self.add_log(type='error', message=str(e), details=details, event_type='vector_store_creation')
            raise

    def system_prompt(self):
        pass

    def user_prompt(self):
        pass

    def _load_few_shot_examples(self) -> List[Dict[str, Any]]:
        examples_path = os.path.join(self.data_dir, 'few_shots.jsonl')
        if not os.path.exists(examples_path) or os.path.getsize(examples_path) == 0:
            return []

        examples = []
        with open(examples_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        output_str = json.dumps({
                            "answer": data["output"]["answer"],
                            "tone": data["output"]["tone"],
                            "actions": data["output"]["actions"]
                        }, ensure_ascii=False)
                        examples.append({
                            "input": data["input"],
                            "context": data.get("context", ""),
                            "output": data["output"]  # Единое поле для ответа
                        })

                        '''examples.append({
                            "input": data["input"],
                            "context": data.get("context", ""),
                            "answer": data["output"]["answer"],
                            "tone": data["output"]["tone"],
                            "actions": json.dumps(data["output"]["actions"], ensure_ascii=False)
                        })'''
                    except Exception as e:
                        self.add_log(
                            type='error',
                            message=f"Ошибка парсинга few-shot примера: {str(e)}",
                            event='few_shot_error'
                        )
        return examples

    def _init_example_selector(self) -> Optional[SemanticSimilarityExampleSelector]:
        if not self.few_shot_examples:
            return None

        try:
            embedding_kwargs = {
                "api_key": self.model_config.api_key,
                "model": self.model_config.embedding_model
            }
            if self.model_config.base_url:
                embedding_kwargs["base_url"] = self.model_config.base_url

            embeddings = OpenAIEmbeddings(**embedding_kwargs)

            example_texts = [
                f"Вопрос: {ex['input']}\nКонтекст: {ex['context']}"
                for ex in self.few_shot_examples
            ]

            metadatas = []
            for ex in self.few_shot_examples:
                metadatas.append({
                    "input": ex["input"],
                    "output": json.dumps(ex["output"], ensure_ascii=False),
                    "context": ex.get("context", "")
                })

            vectorstore = FAISS.from_texts(
                example_texts,
                embeddings,
                metadatas=metadatas
            )

            self.add_log(
                event='few_shot_selector_initialized',
                example_count=len(self.few_shot_examples),
                model=self.model_config.embedding_model
            )

            return SemanticSimilarityExampleSelector(
                vectorstore=vectorstore,
                k=min(2, len(self.few_shot_examples)),
                input_keys=["input"],
            )

        except Exception as e:
            self.add_log(
                type='error',
                message=f"Ошибка инициализации few-shot селектора: {str(e)}",
                event='few_shot_init_error'
            )
            return None

    def retrieval_chain(self, model: ChatOpenAI, vector_store: FAISS):
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        chain_config = self._prompt_config["prompts"]["support_answer"]
        current_version = chain_config["current_version"]
        version_config = chain_config["versions"][current_version]

        messages = []
        if system_template:= version_config.get('system'):
            system_prompt = SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=system_template
                )
            )
            #self.add_log(message=system_prompt.prompt)
            messages.append(system_prompt)

        if self.example_selector:
            example_prompt = ChatPromptTemplate.from_messages([
                ("human", "{input}"),
                ("ai", "{output}")
            ])

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_selector=self.example_selector,
                example_prompt=example_prompt,
                input_variables=["input"]
            )
            #self.add_log(message=few_shot_prompt.prompt)
            messages.append(few_shot_prompt)

        # Human промпт с вопросом
        user_template = version_config['user']
        user_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=user_template
            )
        )
        #self.add_log(event='human_message_prompt_t', message=user_prompt.prompt)
        messages.append(user_prompt)

        input_vars = version_config["input_variables"].copy()

        self.add_log(event='awaiting_variables', message=input_vars)
        qa_prompt = ChatPromptTemplate(
            messages=messages,
            input_variables=input_vars
        )

        document_chain = create_stuff_documents_chain(
            model.with_structured_output(FAQResponse, method="function_calling", include_raw=False),
            qa_prompt
        )
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        try:
            input_schema = retrieval_chain.get_input_schema()
            input_keys = list(input_schema.model_fields.keys())
            self.add_log(event="chain_created", input_keys=input_keys)
        except Exception as e:
            self.add_log(type='warning', message=f"Не удалось определить входные ключи цепочки: {str(e)}")
        return retrieval_chain

    def faq_processor(self, query: str, retrieval_chain):
        try:
            with get_openai_callback() as cb:
                full_response = retrieval_chain.invoke({
                    "brand_name": self.model_config.brand_name,
                    "input": query,
                    "history": self.format_conversation_history()
                })

                response: FAQResponse = full_response["answer"]

                if not response.actions or len(response.actions) == 0:
                    self.add_log(
                        type='warning',
                        query=query,
                        message="actions был пуст, применен fallback",
                        another_actions=json.dumps(response.model_dump_json(), ensure_ascii=False)
                    )
                    response.actions = [
                        "Задайте уточняющий вопрос",
                        "Обратитесь в службу поддержки"
                    ]


                serializable_response = {
                    "answer": response.answer,
                    "tone": response.tone,
                    "actions": response.actions,
                    "usage": {"total_tokens": cb.total_tokens, "prompt_tokens": cb.prompt_tokens,
                              "completion_tokens": cb.completion_tokens}
                }

            self.add_log(query=query, message=serializable_response)
            return response
        except Exception as e:
            error_context = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "chain_input_keys": retrieval_chain.input_keys if hasattr(retrieval_chain, 'input_keys') else None,
                "query": query,
                "history": self.format_conversation_history()
            }
            self.add_log(type='error', message="Chain execution failed", **error_context)
            print("Произошла ошибка при обработке запроса. Попробуйте еще раз.")

            return FAQResponse(
                answer="Произошла ошибка при обработке запроса. Попробуйте еще раз.",
                tone="ошибка",
                actions=[]
            )


    def orders_processor(self, query: str):
        match = re.fullmatch(r'/order\s+(\d+)', query.strip())
        if not match:
            response = 'Неверный формат. Используйте: /order <номер>'
            self.add_log(type='error', query=query, message=response, event='order_error', event_type='invalid_format')
            return response

        order_id = match.group(1)
        if order := self.orders.get(order_id):
            response = f'Заказ #{match.group(1)}: {format_order_details(order)}'
            self.add_log(query=query, message=response)
            return response
        else:
            response = 'Пожалуйста, проверьте введенные данные.'
            self.add_log(type='error', query=query, message=response, event='order_error', event_type='not_found')
            return response

    def add_log(self, type: str = "info", query: str = None, message: Union[str, dict] = None, **kwargs):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "message": message,
            **kwargs
        }
        if type == 'error':
            self.logger.error(json.dumps(log_entry, ensure_ascii=False))
        else:
            self.logger.info(json.dumps(log_entry, ensure_ascii=False))


def format_order_details(info: dict) -> str:
    status = info.get("status")
    if status == "in_transit":
        eta = info.get("eta_days", 0)
        carrier = info.get("carrier", "неизвестен")
        detail = f"Заказ в пути. Ожидаемая доставка через {eta} дн. Перевозчик: {carrier}."
    elif status == "delivered":
        delivered_at = info.get("delivered_at", "не указана")
        try:
            # Опционально: можно отформатировать дату красиво
            date_obj = datetime.strptime(delivered_at, "%Y-%m-%d")
            delivered_at = date_obj.strftime("%d.%m.%Y")
        except ValueError:
            pass  # оставляем как есть, если не удалось распарсить
        detail = f"Заказ доставлен {delivered_at}."
    elif status == "processing":
        note = info.get("note", "Без примечаний")
        detail = f"Заказ в обработке. {note}"
    else:
        detail = f"Статус заказа: {status}." if status else "Информация о заказе недоступна."

    return detail


def main():
    parser = argparse.ArgumentParser(description="Consultant Bot")
    parser.add_argument('--url', type=str, help='Base URL for LLM API')
    parser.add_argument('--model', type=str, help='LLM model name')
    parser.add_argument('--api-key', type=str, help='API key for authentication')
    args = parser.parse_args()

    try:
        model_config = setup_api_config()
        if args.api_key:
            model_config.api_key = args.api_key
        if args.url:
            model_config.base_url = args.url
        if args.model:
            model_config.llm_model = args.model

        bot = Consultant(model_config=model_config)
        config = bot.model_config
        bot.add_log(event="config_loaded", config=config.to_dict())

        llm_kwargs = {
            "api_key": config.api_key,
            "temperature": config.temperature,
            "model_name": config.llm_model,
            "openai_api_base": config.base_url}
        model = ChatOpenAI(**llm_kwargs)

        # Создаем векторное хранилище
        vector_store = bot.create_vector_store()
        retrieval_chain = bot.retrieval_chain(model=model, vector_store=vector_store)

        # Получаем запрос от пользователя
        print("\n" + "=" * 50)
        print("Введите 'exit' для выхода")

        while True:
            query = input("\nВаш вопрос: ").strip()
            bot.add_to_history("user", query)
            if query.lower() in ['exit', 'quit', 'выйти']:
                print("До свидания! 🐱")
                bot.add_log(message="Пользователь инициировал выход.")
                break

            if not query:
                continue

            if query.startswith("/order"):
                response = bot.orders_processor(query=query)
                print(response)
                bot.add_to_history("assistant", response)
                continue

            response = bot.faq_processor(query=query, retrieval_chain=retrieval_chain)
            print(response.answer)
            bot.add_to_history("assistant", response.answer)

    except Exception as e:
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "critical_error",
            "error": str(e),
            "traceback": traceback.format_exc() if 'traceback' in sys.modules else None
        }
        logging.getLogger(__name__).error(json.dumps(error_entry, ensure_ascii=False))
        print("Критическая ошибка приложения. Детали записаны в лог.")
        exit(1)


if __name__ == "__main__":
    main()