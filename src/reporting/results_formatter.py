"""Форматировщик результатов экспериментов для RL агентов.

Этот модуль предоставляет класс ResultsFormatter для создания
комплексных отчетов по результатам обучения и оценки агентов
в различных форматах (HTML, Markdown, LaTeX, JSON, CSV).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template

from src.evaluation.evaluator import EvaluationMetrics
from src.utils.rl_logging import get_logger

logger = get_logger(__name__)


class ReportConfig:
    """Конфигурация для генерации отчетов."""

    def __init__(
        self,
        language: str = "ru",
        theme: str = "default",
        include_plots: bool = True,
        include_statistics: bool = True,
        decimal_places: int = 4,
        date_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        """Инициализация конфигурации отчета.

        Args:
            language: Язык отчета ('ru' или 'en').
            theme: Тема оформления ('default', 'dark', 'minimal').
            include_plots: Включать ли графики в отчет.
            include_statistics: Включать ли статистику в отчет.
            decimal_places: Количество знаков после запятой.
            date_format: Формат даты и времени.
        """
        self.language = language
        self.theme = theme
        self.include_plots = include_plots
        self.include_statistics = include_statistics
        self.decimal_places = decimal_places
        self.date_format = date_format


class ResultsFormatter:
    """Форматировщик результатов экспериментов RL агентов.

    Поддерживает генерацию отчетов в различных форматах:
    - HTML с интерактивными элементами
    - Markdown для документации
    - LaTeX для научных публикаций
    - JSON для программного анализа
    - CSV для табличных данных
    """

    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        config: Optional[ReportConfig] = None,
    ) -> None:
        """Инициализация форматировщика результатов.

        Args:
            templates_dir: Директория с шаблонами отчетов.
            output_dir: Директория для сохранения отчетов.
            config: Конфигурация отчетов.
        """
        self.templates_dir = templates_dir or Path(__file__).parent / "templates"
        self.output_dir = output_dir or Path("results/reports")
        self.config = config or ReportConfig()

        # Создаем директории если они не существуют
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Настраиваем Jinja2
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True,
        )

        # Добавляем пользовательские фильтры
        self._setup_jinja_filters()

        # Словари для локализации
        self.translations = self._load_translations()

        logger.info(
            f"Инициализирован ResultsFormatter: templates_dir={self.templates_dir}, "
            f"output_dir={self.output_dir}, language={self.config.language}"
        )

    def _setup_jinja_filters(self) -> None:
        """Настройка пользовательских фильтров для Jinja2."""

        def format_number(value: float, places: Optional[int] = None) -> str:
            """Форматирование числа с заданным количеством знаков."""
            places = places or self.config.decimal_places
            return f"{value:.{places}f}"

        def format_percentage(value: float, places: int = 2) -> str:
            """Форматирование процентов."""
            return f"{value * 100:.{places}f}%"

        def format_datetime(value: datetime) -> str:
            """Форматирование даты и времени."""
            return value.strftime(self.config.date_format)

        def translate(key: str) -> str:
            """Перевод текста."""
            return self.translations.get(self.config.language, {}).get(key, key)

        # Регистрируем фильтры
        self.jinja_env.filters["format_number"] = format_number
        self.jinja_env.filters["format_percentage"] = format_percentage
        self.jinja_env.filters["format_datetime"] = format_datetime
        self.jinja_env.filters["translate"] = translate

    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Загрузка переводов для интернационализации."""
        return {
            "ru": {
                "title": "Отчет по результатам обучения RL агентов",
                "agent": "Агент",
                "environment": "Среда",
                "reward": "Награда",
                "episode_length": "Длина эпизода",
                "success_rate": "Процент успеха",
                "training_time": "Время обучения",
                "evaluation_results": "Результаты оценки",
                "statistics": "Статистика",
                "comparison": "Сравнение",
                "experiment": "Эксперимент",
                "summary": "Сводка",
                "mean": "Среднее",
                "std": "Стандартное отклонение",
                "min": "Минимум",
                "max": "Максимум",
                "median": "Медиана",
                "generated_at": "Сгенерировано",
            },
            "en": {
                "title": "RL Agent Training Results Report",
                "agent": "Agent",
                "environment": "Environment",
                "reward": "Reward",
                "episode_length": "Episode Length",
                "success_rate": "Success Rate",
                "training_time": "Training Time",
                "evaluation_results": "Evaluation Results",
                "statistics": "Statistics",
                "comparison": "Comparison",
                "experiment": "Experiment",
                "summary": "Summary",
                "mean": "Mean",
                "std": "Standard Deviation",
                "min": "Minimum",
                "max": "Maximum",
                "median": "Median",
                "generated_at": "Generated at",
            },
        }

    def format_single_agent_report(
        self,
        agent_name: str,
        evaluation_results: EvaluationMetrics,
        quantitative_results: Optional[Any] = None,
        output_format: str = "html",
        filename: Optional[str] = None,
    ) -> Path:
        """Создание отчета по одному агенту.

        Args:
            agent_name: Название агента.
            evaluation_results: Результаты оценки агента.
            quantitative_results: Количественные результаты оценки.
            output_format: Формат вывода ('html', 'markdown', 'latex', 'json').
            filename: Имя файла для сохранения.

        Returns:
            Путь к созданному файлу отчета.
        """
        logger.info(f"Создание отчета по агенту {agent_name}")

        # Подготавливаем данные для шаблона
        template_data = {
            "agent_name": agent_name,
            "evaluation_results": evaluation_results,
            "quantitative_results": quantitative_results,
            "config": self.config,
            "generated_at": datetime.now(),
            "report_type": "single_agent",
        }

        # Добавляем статистику если есть количественные результаты
        if quantitative_results:
            template_data["statistics"] = self._calculate_statistics(
                quantitative_results
            )

        # Генерируем отчет
        return self._generate_report(
            template_data=template_data,
            template_name=f"single_agent.{output_format}",
            output_format=output_format,
            filename=filename or f"{agent_name}_report",
        )

    def format_comparison_report(
        self,
        agents_results: Dict[str, EvaluationMetrics],
        quantitative_results: Optional[Dict[str, Any]] = None,
        output_format: str = "html",
        filename: Optional[str] = None,
    ) -> Path:
        """Создание сравнительного отчета нескольких агентов.

        Args:
            agents_results: Результаты оценки агентов.
            quantitative_results: Количественные результаты агентов.
            output_format: Формат вывода.
            filename: Имя файла для сохранения.

        Returns:
            Путь к созданному файлу отчета.
        """
        logger.info(f"Создание сравнительного отчета для {len(agents_results)} агентов")

        # Подготавливаем данные для сравнения
        comparison_data = self._prepare_comparison_data(
            agents_results, quantitative_results
        )

        template_data = {
            "agents_results": agents_results,
            "quantitative_results": quantitative_results,
            "comparison_data": comparison_data,
            "config": self.config,
            "generated_at": datetime.now(),
            "report_type": "comparison",
        }

        return self._generate_report(
            template_data=template_data,
            template_name=f"comparison.{output_format}",
            output_format=output_format,
            filename=filename or "agents_comparison",
        )

    def format_experiment_report(
        self,
        experiment_name: str,
        experiment_data: Dict[str, Any],
        output_format: str = "html",
        filename: Optional[str] = None,
    ) -> Path:
        """Создание отчета по эксперименту.

        Args:
            experiment_name: Название эксперимента.
            experiment_data: Данные эксперимента.
            output_format: Формат вывода.
            filename: Имя файла для сохранения.

        Returns:
            Путь к созданному файлу отчета.
        """
        logger.info(f"Создание отчета по эксперименту {experiment_name}")

        template_data = {
            "experiment_name": experiment_name,
            "experiment_data": experiment_data,
            "config": self.config,
            "generated_at": datetime.now(),
            "report_type": "experiment",
        }

        return self._generate_report(
            template_data=template_data,
            template_name=f"experiment.{output_format}",
            output_format=output_format,
            filename=filename or f"{experiment_name}_report",
        )

    def format_summary_report(
        self,
        experiments_data: Dict[str, Dict[str, Any]],
        output_format: str = "html",
        filename: Optional[str] = None,
    ) -> Path:
        """Создание сводного отчета по всем экспериментам.

        Args:
            experiments_data: Данные всех экспериментов.
            output_format: Формат вывода.
            filename: Имя файла для сохранения.

        Returns:
            Путь к созданному файлу отчета.
        """
        logger.info(
            f"Создание сводного отчета по {len(experiments_data)} экспериментам"
        )

        # Подготавливаем сводную статистику
        summary_stats = self._calculate_summary_statistics(experiments_data)

        template_data = {
            "experiments_data": experiments_data,
            "summary_stats": summary_stats,
            "config": self.config,
            "generated_at": datetime.now(),
            "report_type": "summary",
        }

        return self._generate_report(
            template_data=template_data,
            template_name=f"summary.{output_format}",
            output_format=output_format,
            filename=filename or "experiments_summary",
        )

    def export_to_csv(
        self,
        data: Union[EvaluationMetrics, Dict[str, EvaluationMetrics]],
        filename: Optional[str] = None,
    ) -> Path:
        """Экспорт данных в CSV формат.

        Args:
            data: Данные для экспорта.
            filename: Имя файла для сохранения.

        Returns:
            Путь к созданному CSV файлу.
        """
        logger.info("Экспорт данных в CSV формат")

        # Преобразуем данные в DataFrame
        if isinstance(data, dict):
            # Множественные агенты
            df_data = []
            for agent_name, results in data.items():
                row = {"agent": agent_name}
                row.update(self._evaluation_results_to_dict(results))
                df_data.append(row)
            df = pd.DataFrame(df_data)
        else:
            # Один агент
            df = pd.DataFrame([self._evaluation_results_to_dict(data)])

        # Сохраняем в CSV
        output_path = self.output_dir / f"{filename or 'results'}.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"CSV файл сохранен: {output_path}")
        return output_path

    def export_to_json(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """Экспорт данных в JSON формат.

        Args:
            data: Данные для экспорта.
            filename: Имя файла для сохранения.

        Returns:
            Путь к созданному JSON файлу.
        """
        logger.info("Экспорт данных в JSON формат")

        # Преобразуем данные в JSON-сериализуемый формат
        json_data = self._prepare_json_data(data)

        # Сохраняем в JSON
        output_path = self.output_dir / f"{filename or 'results'}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"JSON файл сохранен: {output_path}")
        return output_path

    def _generate_report(
        self,
        template_data: Dict[str, Any],
        template_name: str,
        output_format: str,
        filename: str,
    ) -> Path:
        """Генерация отчета из шаблона.

        Args:
            template_data: Данные для шаблона.
            template_name: Имя шаблона.
            output_format: Формат вывода.
            filename: Имя файла.

        Returns:
            Путь к созданному файлу.
        """
        try:
            # Загружаем шаблон
            template = self.jinja_env.get_template(template_name)
        except Exception:
            # Если шаблон не найден, создаем базовый
            logger.warning(f"Шаблон {template_name} не найден, используем базовый")
            template = self._create_default_template(output_format)

        # Рендерим шаблон
        rendered_content = template.render(**template_data)

        # Сохраняем файл
        output_path = self.output_dir / f"{filename}.{output_format}"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered_content)

        logger.info(f"Отчет сохранен: {output_path}")
        return output_path

    def _create_default_template(self, output_format: str) -> Template:
        """Создание базового шаблона для формата."""
        if output_format == "html":
            template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Отчет по результатам обучения RL агентов</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric { margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Отчет по результатам обучения RL агентов</h1>
    <p>Сгенерировано: {{ generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
    
    {% if agent_name %}
        <h2>Агент: {{ agent_name }}</h2>
    {% endif %}
    
    {% if evaluation_results %}
        <h3>Результаты оценки</h3>
        <div class="metric">
            Средняя награда: {{ "%.4f"|format(evaluation_results.mean_reward) }}
        </div>
        <div class="metric">
            Стандартное отклонение награды: {{ "%.4f"|format(evaluation_results.std_reward) }}
        </div>
        <div class="metric">
            Средняя длина эпизода: {{ "%.4f"|format(evaluation_results.mean_episode_length) }}
        </div>
        <div class="metric">
            Процент успеха: {{ "%.2f%%"|format(evaluation_results.success_rate * 100) }}
        </div>
    {% endif %}
</body>
</html>
            """
        elif output_format == "markdown":
            template_content = """
# Отчет по результатам обучения RL агентов

Сгенерировано: {{ generated_at.strftime('%Y-%m-%d %H:%M:%S') }}

{% if agent_name %}
## Агент: {{ agent_name }}
{% endif %}

{% if evaluation_results %}
### Результаты оценки

- Средняя награда: {{ "%.4f"|format(evaluation_results.mean_reward) }}
- Стандартное отклонение награды: {{ "%.4f"|format(evaluation_results.std_reward) }}
- Средняя длина эпизода: {{ "%.4f"|format(evaluation_results.mean_episode_length) }}
- Процент успеха: {{ "%.2f%%"|format(evaluation_results.success_rate * 100) }}
{% endif %}
            """
        else:
            template_content = "{{ data | tojson(indent=2) }}"

        return self.jinja_env.from_string(template_content)

    def _calculate_statistics(self, quantitative_results: Any) -> Dict[str, Any]:
        """Расчет статистики по количественным результатам."""
        stats = {}

        if hasattr(quantitative_results, "rewards") and quantitative_results.rewards:
            rewards = quantitative_results.rewards
            stats["reward"] = {
                "mean": float(pd.Series(rewards).mean()),
                "std": float(pd.Series(rewards).std()),
                "min": float(pd.Series(rewards).min()),
                "max": float(pd.Series(rewards).max()),
                "median": float(pd.Series(rewards).median()),
            }

        return stats

    def _prepare_comparison_data(
        self,
        agents_results: Dict[str, EvaluationMetrics],
        quantitative_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Подготовка данных для сравнения агентов."""
        comparison_data = {
            "agents": list(agents_results.keys()),
            "metrics": {},
        }

        # Собираем метрики для сравнения
        for agent_name, results in agents_results.items():
            if "mean_reward" not in comparison_data["metrics"]:
                comparison_data["metrics"]["mean_reward"] = {}
            comparison_data["metrics"]["mean_reward"][agent_name] = results.mean_reward

        return comparison_data

    def _calculate_summary_statistics(
        self, experiments_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Расчет сводной статистики по экспериментам."""
        summary = {
            "total_experiments": len(experiments_data),
            "experiments": list(experiments_data.keys()),
        }

        return summary

    def _evaluation_results_to_dict(self, results: EvaluationMetrics) -> Dict[str, Any]:
        """Преобразование результатов оценки в словарь."""
        return {
            "mean_reward": results.mean_reward,
            "std_reward": results.std_reward,
            "min_reward": results.min_reward,
            "max_reward": results.max_reward,
            "mean_episode_length": results.mean_episode_length,
            "std_episode_length": results.std_episode_length,
            "min_episode_length": results.min_episode_length,
            "max_episode_length": results.max_episode_length,
            "success_rate": results.success_rate,
            "total_episodes": results.total_episodes,
            "total_timesteps": results.total_timesteps,
            "evaluation_time": results.evaluation_time,
        }

    def _prepare_json_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Подготовка данных для JSON сериализации."""
        json_data = {}

        for key, value in data.items():
            if hasattr(value, "__dict__"):
                # Объект с атрибутами
                json_data[key] = vars(value)
            elif isinstance(value, dict):
                # Рекурсивно обрабатываем словари
                json_data[key] = self._prepare_json_data(value)
            else:
                json_data[key] = value

        return json_data
