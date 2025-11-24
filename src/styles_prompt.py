from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import yaml

#required_keys = {'brand', 'tone.persona', 'format.fields.answer'}
class MissingConfigKeyError(Exception):
    """Ошибка при отсутствии критичного ключа в конфиге."""
    pass

@dataclass
class Style:
    brand: str
    persona: str
    sentences_max: int
    bullets: bool
    avoid: List[str]
    must_include: List[str]
    fb_no_data: str
    fields_answer: str
    fields_tone: str
    fields_actions: str

    @classmethod
    def from_dict(cls, data: dict, required_keys: Set[str] = None) -> "Style":
        """
        Парсинг из dict с опциональной валидацией ключей.

        Args:
            data: Словарь из YAML
            required_keys: Набор обязательных ключей.
                          Формат: 'brand', 'tone.persona', 'format.fields.answer'
        """

        def get_nested(d: dict, path: str, default=None):
            """Получить значение по пути вида 'tone.persona'."""
            keys = path.split('.')
            value = d
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return default
            return value if value is not None else default

        # Проверка обязательных ключей
        if required_keys:
            missing = [key for key in required_keys if not get_nested(data, key)]
            if missing:
                raise MissingConfigKeyError(
                    f"Отсутствуют обязательные ключи: {', '.join(missing)}"
                )

        tone = data.get('tone', {})
        fields = data.get('format', {}).get('fields', {})
        fallback = data.get('fallback', {})

        return cls(
            brand=data.get('brand'),
            persona=tone.get('persona'),
            sentences_max=int(tone.get('sentences_max', 3)),
            bullets=bool(tone.get('bullets', False)),
            avoid=tone.get('avoid', []),
            must_include=tone.get('must_include', []),
            fb_no_data=fallback.get('no_data', 'Информация недоступна'),
            fields_answer=fields.get('answer'),
            fields_tone=fields.get('tone'),
            fields_actions=fields.get('actions'),
        )

class StyleParser:
    def __init__(self):
        self.config_path = Path(__file__).parent.parent / "data" / "style_guide.yaml"
        self._style: Optional[Style] = None

    @property
    def style(self) -> Style:
        """Ленивая загрузка с валидацией."""
        if self._style is None:
            try:
                with open(self.config_path, encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                # Валидируем обязательные ключи
                self._style = Style.from_dict(config)
            except MissingConfigKeyError as e:
                print(f"❌ Ошибка конфига: {e}")
                raise
            except FileNotFoundError:
                print(f"❌ Файл конфига не найден: {self.config_path}")
                raise
        return self._style

    def style_prompt(self) -> str:
        """Генерирует промпт с динамической нумерацией."""
        s = self.style

        # Список всех пунктов с условиями отображения
        sections = [
            (s.persona and s.sentences_max,
             f"Тон: {s.persona}. Максимум {s.sentences_max} предложений в ответе."),

            (s.bullets is not None,
             f"Форматирование: {'используй' if s.bullets else 'не используй'} маркированные списки."),

            (s.avoid,
             f"Избегай: {', '.join(s.avoid)}."),

            (s.must_include,
             f"Обязательно включай: {', '.join(s.must_include)}."),

            (s.fb_no_data,
             f"При отсутствии данных: «{s.fb_no_data}»."),

            (s.fields_answer and s.fields_tone and s.fields_actions,
             f"Формат ответа (JSON):\n{{\n"
             f'  "answer": "{s.fields_answer}",\n'
             f'  "tone": "{s.fields_tone}",\n'
             f'  "actions": "{s.fields_actions}"\n'
             f"}}"),
        ]

        # Фильтруем и нумеруем только заполненные пункты
        numbered_sections = [
            f"{i}. {text}"
            for i, (condition, text) in enumerate([
                (cond, txt) for cond, txt in sections if cond
                ], start=1)
        ]

        if len(numbered_sections):
            rules_exist_str = ", следуй этим правилам:\n"
        else:
            rules_exist_str = "."

        prompt = (
                f"Ты — консультант магазина {s.brand}{rules_exist_str}" +
                "\n".join(numbered_sections) + "\n"
        )

        return prompt


