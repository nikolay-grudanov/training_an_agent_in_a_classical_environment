"""Система визуализации производительности обучения RL агентов.

Этот модуль предоставляет комплексные инструменты для создания графиков
производительности обучения, включая интерактивные и статические визуализации,
сравнительный анализ агентов, обработку различных источников данных и
настраиваемые стили визуализации.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

from src.utils.logging import get_logger
from src.utils.metrics import MetricsTracker, MetricSummary

# Подавляем предупреждения matplotlib
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

logger = get_logger(__name__)


class PlotStyle:
    """Настройки стилей для графиков."""
    
    # Цветовые палитры
    COLORS_PRIMARY = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    COLORS_SECONDARY = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    COLORS_GRADIENT = ['#440154', '#31688e', '#35b779', '#fde725']
    
    # Стили линий
    LINE_STYLES = ['-', '--', '-.', ':', '-']
    MARKERS = ['o', 's', '^', 'v', 'D']
    
    # Размеры фигур
    FIGSIZE_SMALL = (8, 6)
    FIGSIZE_MEDIUM = (12, 8)
    FIGSIZE_LARGE = (16, 10)
    FIGSIZE_WIDE = (16, 6)
    
    @staticmethod
    def setup_matplotlib_style(style: str = 'seaborn-v0_8') -> None:
        """Настроить стиль matplotlib.
        
        Args:
            style: Название стиля matplotlib
        """
        try:
            plt.style.use(style)
        except OSError:
            # Fallback для старых версий matplotlib
            try:
                plt.style.use('seaborn')
            except OSError:
                # Если и seaborn недоступен, используем default
                plt.style.use('default')
        
        # Настройки по умолчанию
        plt.rcParams.update({
            'figure.figsize': PlotStyle.FIGSIZE_MEDIUM,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'grid.alpha': 0.3,
        })


class DataLoader:
    """Загрузчик данных из различных источников."""
    
    @staticmethod
    def load_from_csv(file_path: Union[str, Path]) -> pd.DataFrame:
        """Загрузить данные из CSV файла.
        
        Args:
            file_path: Путь к CSV файлу
            
        Returns:
            DataFrame с данными метрик
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если формат данных некорректен
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV файл не найден: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Загружены данные из CSV: {file_path} ({len(df)} записей)")
            return df
        except Exception as e:
            raise ValueError(f"Ошибка при загрузке CSV файла: {e}")
    
    @staticmethod
    def load_from_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Загрузить данные из JSON файла.
        
        Args:
            file_path: Путь к JSON файлу
            
        Returns:
            Словарь с данными метрик
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если формат JSON некорректен
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON файл не найден: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Загружены данные из JSON: {file_path}")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка при парсинге JSON файла: {e}")
    
    @staticmethod
    def load_from_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Загрузить данные из JSONL файла (JSON Lines).
        
        Args:
            file_path: Путь к JSONL файлу
            
        Returns:
            Список словарей с данными метрик
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSONL файл не найден: {file_path}")
        
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Ошибка в строке {line_num}: {e}")
            
            logger.info(f"Загружены данные из JSONL: {file_path} ({len(data)} записей)")
            return data
        except Exception as e:
            raise ValueError(f"Ошибка при загрузке JSONL файла: {e}")
    
    @staticmethod
    def load_from_metrics_tracker(tracker: MetricsTracker) -> Dict[str, pd.DataFrame]:
        """Загрузить данные из MetricsTracker.
        
        Args:
            tracker: Экземпляр MetricsTracker
            
        Returns:
            Словарь с DataFrame для каждой метрики
        """
        data = {}
        for metric_name in tracker.metrics:
            points = list(tracker.metrics[metric_name])
            if points:
                df = pd.DataFrame([
                    {
                        'timestep': p.timestep,
                        'episode': p.episode,
                        'value': p.value,
                        'timestamp': p.timestamp,
                        **p.metadata
                    }
                    for p in points
                ])
                data[metric_name] = df
        
        logger.info(f"Загружены данные из MetricsTracker: {len(data)} метрик")
        return data
    
    @staticmethod
    def convert_sb3_logs(log_dir: Union[str, Path]) -> pd.DataFrame:
        """Конвертировать логи Stable-Baselines3 в DataFrame.
        
        Args:
            log_dir: Директория с логами SB3
            
        Returns:
            DataFrame с метриками обучения
        """
        log_dir = Path(log_dir)
        
        # Ищем файлы логов SB3
        progress_files = list(log_dir.glob("**/progress.csv"))
        if not progress_files:
            raise FileNotFoundError(f"Файлы логов SB3 не найдены в {log_dir}")
        
        all_data = []
        for progress_file in progress_files:
            try:
                df = pd.read_csv(progress_file)
                # Добавляем идентификатор эксперимента
                df['experiment'] = progress_file.parent.name
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Ошибка при загрузке {progress_file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Загружены логи SB3: {len(combined_df)} записей")
            return combined_df
        else:
            raise ValueError("Не удалось загрузить ни одного файла логов SB3")


class PerformancePlotter:
    """Основной класс для создания графиков производительности."""
    
    def __init__(
        self,
        style: str = 'seaborn-v0_8',
        color_palette: str = 'husl',
        figsize: Tuple[int, int] = PlotStyle.FIGSIZE_MEDIUM
    ):
        """Инициализация плоттера.
        
        Args:
            style: Стиль matplotlib
            color_palette: Цветовая палитра
            figsize: Размер фигур по умолчанию
        """
        self.style = style
        self.color_palette = color_palette
        self.figsize = figsize
        
        # Настраиваем стили
        PlotStyle.setup_matplotlib_style(style)
        sns.set_palette(color_palette)
        
        logger.info(f"Инициализирован PerformancePlotter (стиль: {style})")
    
    def plot_reward_curve(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        x_col: str = 'timestep',
        y_col: str = 'episode_reward',
        smooth_window: int = 100,
        show_raw: bool = True,
        confidence_interval: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> str:
        """Построить график среднего вознаграждения по временным шагам.
        
        Args:
            data: Данные для построения графика
            x_col: Название колонки для оси X
            y_col: Название колонки для оси Y
            smooth_window: Размер окна для сглаживания
            show_raw: Показать исходные данные
            confidence_interval: Показать доверительный интервал
            save_path: Путь для сохранения
            title: Заголовок графика
            **kwargs: Дополнительные параметры для matplotlib
            
        Returns:
            Путь к сохраненному файлу
        """
        if isinstance(data, dict):
            # Конвертируем из формата MetricsTracker
            if y_col in data:
                df = data[y_col]
            else:
                raise ValueError(f"Метрика {y_col} не найдена в данных")
        else:
            df = data
        
        if df.empty:
            raise ValueError("Данные для построения графика пустые")
        
        # Создаем фигуру
        plt.figure(figsize=self.figsize)
        
        # Исходные данные
        if show_raw:
            y_values = df['value'] if 'value' in df.columns else df[y_col]
            plt.plot(df[x_col], y_values, alpha=0.3, color='lightblue', 
                    label='Исходные данные', **kwargs)
        
        # Сглаженные данные
        if len(df) >= smooth_window:
            y_values = df['value'] if 'value' in df.columns else df[y_col]
            smoothed = pd.Series(y_values).rolling(window=smooth_window, center=True).mean()
            plt.plot(df[x_col], smoothed, linewidth=2, color='darkblue',
                    label=f'Скользящее среднее ({smooth_window})', **kwargs)
            
            # Доверительный интервал
            if confidence_interval and len(df) >= smooth_window * 2:
                smoothed_std = pd.Series(y_values).rolling(window=smooth_window, center=True).std()
                upper = smoothed + smoothed_std
                lower = smoothed - smoothed_std
                plt.fill_between(df[x_col], lower, upper, alpha=0.2, color='darkblue',
                               label='Доверительный интервал (±1σ)')
        
        # Оформление
        plt.xlabel('Временные шаги')
        plt.ylabel('Вознаграждение за эпизод')
        plt.title(title or 'Кривая обучения: Вознаграждение по времени')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Сохранение
        if save_path is None:
            save_path = Path('results/plots/reward_curve.png')
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График вознаграждения сохранен: {save_path}")
        return str(save_path)
    
    def plot_episode_lengths(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        x_col: str = 'episode',
        y_col: str = 'episode_length',
        smooth_window: int = 50,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> str:
        """Построить график длины эпизодов.
        
        Args:
            data: Данные для построения графика
            x_col: Название колонки для оси X
            y_col: Название колонки для оси Y
            smooth_window: Размер окна для сглаживания
            save_path: Путь для сохранения
            title: Заголовок графика
            **kwargs: Дополнительные параметры
            
        Returns:
            Путь к сохраненному файлу
        """
        if isinstance(data, dict):
            if y_col in data:
                df = data[y_col]
            else:
                raise ValueError(f"Метрика {y_col} не найдена в данных")
        else:
            df = data
        
        if df.empty:
            raise ValueError("Данные для построения графика пустые")
        
        plt.figure(figsize=self.figsize)
        
        # Исходные данные
        y_values = df['value'] if 'value' in df.columns else df[y_col]
        plt.plot(df[x_col], y_values, alpha=0.5, color='green', 
                label='Длина эпизода', **kwargs)
        
        # Сглаженные данные
        if len(df) >= smooth_window:
            smoothed = pd.Series(y_values).rolling(window=smooth_window, center=True).mean()
            plt.plot(df[x_col], smoothed, linewidth=2, color='darkgreen',
                    label=f'Скользящее среднее ({smooth_window})')
        
        # Оформление
        plt.xlabel('Номер эпизода')
        plt.ylabel('Длина эпизода (шаги)')
        plt.title(title or 'Длина эпизодов в процессе обучения')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Сохранение
        if save_path is None:
            save_path = Path('results/plots/episode_lengths.png')
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График длины эпизодов сохранен: {save_path}")
        return str(save_path)
    
    def plot_loss_curves(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        loss_columns: Optional[List[str]] = None,
        x_col: str = 'timestep',
        smooth_window: int = 100,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> str:
        """Построить график функций потерь.
        
        Args:
            data: Данные для построения графика
            loss_columns: Список колонок с потерями
            x_col: Название колонки для оси X
            smooth_window: Размер окна для сглаживания
            save_path: Путь для сохранения
            title: Заголовок графика
            **kwargs: Дополнительные параметры
            
        Returns:
            Путь к сохраненному файлу
        """
        if loss_columns is None:
            loss_columns = ['training_loss', 'policy_loss', 'value_loss', 'entropy_loss']
        
        if isinstance(data, dict):
            # Объединяем данные из разных метрик
            combined_data = []
            for loss_col in loss_columns:
                if loss_col in data:
                    df_loss = data[loss_col].copy()
                    df_loss['loss_type'] = loss_col
                    combined_data.append(df_loss)
            
            if not combined_data:
                raise ValueError("Ни одна из указанных метрик потерь не найдена")
            
            df = pd.concat(combined_data, ignore_index=True)
        else:
            df = data
        
        plt.figure(figsize=self.figsize)
        
        # Строим график для каждого типа потерь
        colors = PlotStyle.COLORS_PRIMARY
        for i, loss_col in enumerate(loss_columns):
            if isinstance(data, dict):
                if loss_col not in data:
                    continue
                loss_data = data[loss_col]
            else:
                if loss_col not in df.columns:
                    continue
                loss_data = df[df['loss_type'] == loss_col] if 'loss_type' in df.columns else df
            
            if loss_data.empty:
                continue
            
            color = colors[i % len(colors)]
            
            # Исходные данные
            plt.plot(loss_data[x_col], loss_data['value'], alpha=0.3, color=color)
            
            # Сглаженные данные
            if len(loss_data) >= smooth_window:
                smoothed = pd.Series(loss_data['value']).rolling(window=smooth_window, center=True).mean()
                plt.plot(loss_data[x_col], smoothed, linewidth=2, color=color,
                        label=loss_col.replace('_', ' ').title())
        
        # Оформление
        plt.xlabel('Временные шаги')
        plt.ylabel('Значение функции потерь')
        plt.title(title or 'Кривые обучения: Функции потерь')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Логарифмическая шкала для потерь
        
        # Сохранение
        if save_path is None:
            save_path = Path('results/plots/loss_curves.png')
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График функций потерь сохранен: {save_path}")
        return str(save_path)
    
    def plot_multiple_agents(
        self,
        agents_data: Dict[str, Union[pd.DataFrame, Dict[str, Any]]],
        metric: str = 'episode_reward',
        x_col: str = 'timestep',
        smooth_window: int = 100,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> str:
        """Построить сравнительный график нескольких агентов.
        
        Args:
            agents_data: Словарь с данными агентов {имя_агента: данные}
            metric: Название метрики для сравнения
            x_col: Название колонки для оси X
            smooth_window: Размер окна для сглаживания
            save_path: Путь для сохранения
            title: Заголовок графика
            **kwargs: Дополнительные параметры
            
        Returns:
            Путь к сохраненному файлу
        """
        plt.figure(figsize=self.figsize)
        
        colors = PlotStyle.COLORS_PRIMARY
        line_styles = PlotStyle.LINE_STYLES
        
        for i, (agent_name, agent_data) in enumerate(agents_data.items()):
            # Получаем данные для агента
            if isinstance(agent_data, dict):
                if metric in agent_data:
                    df = agent_data[metric]
                else:
                    logger.warning(f"Метрика {metric} не найдена для агента {agent_name}")
                    continue
            else:
                df = agent_data
            
            if df.empty:
                logger.warning(f"Пустые данные для агента {agent_name}")
                continue
            
            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)]
            
            # Исходные данные (полупрозрачные)
            plt.plot(df[x_col], df['value'], alpha=0.2, color=color, linestyle=line_style)
            
            # Сглаженные данные
            if len(df) >= smooth_window:
                smoothed = df['value'].rolling(window=smooth_window, center=True).mean()
                plt.plot(df[x_col], smoothed, linewidth=2, color=color, 
                        linestyle=line_style, label=agent_name)
        
        # Оформление
        plt.xlabel('Временные шаги')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title or f'Сравнение агентов: {metric.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Сохранение
        if save_path is None:
            save_path = Path(f'results/plots/comparison_{metric}.png')
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Сравнительный график агентов сохранен: {save_path}")
        return str(save_path)
    
    def create_dashboard(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ) -> str:
        """Создать дашборд с несколькими графиками.
        
        Args:
            data: Данные для построения графиков
            metrics: Список метрик для отображения
            save_path: Путь для сохранения
            title: Заголовок дашборда
            
        Returns:
            Путь к сохраненному файлу
        """
        if metrics is None:
            if isinstance(data, dict):
                metrics = list(data.keys())[:4]  # Первые 4 метрики
            else:
                # Пытаемся найти стандартные метрики
                available_cols = data.columns.tolist()
                metrics = [col for col in ['episode_reward', 'episode_length', 'training_loss'] 
                          if col in available_cols][:4]
        
        n_metrics = len(metrics)
        if n_metrics == 0:
            raise ValueError("Нет доступных метрик для отображения")
        
        # Определяем размер сетки
        if n_metrics == 1:
            rows, cols = 1, 1
        elif n_metrics == 2:
            rows, cols = 1, 2
        elif n_metrics <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Строим графики для каждой метрики
        for i, metric in enumerate(metrics[:len(axes)]):
            ax = axes[i]
            
            # Получаем данные
            if isinstance(data, dict):
                if metric in data:
                    df = data[metric]
                else:
                    ax.text(0.5, 0.5, f'Метрика {metric}\nне найдена', 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
            else:
                df = data
            
            if df.empty:
                ax.text(0.5, 0.5, f'Нет данных\nдля {metric}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Строим график
            x_col = 'timestep' if 'timestep' in df.columns else df.columns[0]
            y_col = 'value' if 'value' in df.columns else metric
            
            if y_col in df.columns:
                ax.plot(df[x_col], df[y_col], alpha=0.7, linewidth=1)
                
                # Добавляем сглаживание если данных достаточно
                if len(df) >= 50:
                    smoothed = pd.Series(df[y_col]).rolling(window=50, center=True).mean()
                    ax.plot(df[x_col], smoothed, linewidth=2)
            
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Временные шаги')
            ax.set_ylabel('Значение')
            ax.grid(True, alpha=0.3)
        
        # Скрываем лишние подграфики
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title or 'Дашборд метрик обучения', fontsize=16)
        plt.tight_layout()
        
        # Сохранение
        if save_path is None:
            save_path = Path('results/plots/training_dashboard.png')
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Дашборд метрик сохранен: {save_path}")
        return str(save_path)


class InteractivePlotter:
    """Класс для создания интерактивных графиков с Plotly."""
    
    def __init__(self, theme: str = 'plotly_white'):
        """Инициализация интерактивного плоттера.
        
        Args:
            theme: Тема Plotly
        """
        self.theme = theme
        logger.info(f"Инициализирован InteractivePlotter (тема: {theme})")
    
    def plot_interactive_reward_curve(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        x_col: str = 'timestep',
        y_col: str = 'episode_reward',
        smooth_window: int = 100,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ) -> str:
        """Создать интерактивный график вознаграждения.
        
        Args:
            data: Данные для построения графика
            x_col: Название колонки для оси X
            y_col: Название колонки для оси Y
            smooth_window: Размер окна для сглаживания
            save_path: Путь для сохранения HTML файла
            title: Заголовок графика
            
        Returns:
            Путь к сохраненному HTML файлу
        """
        if isinstance(data, dict):
            if y_col in data:
                df = data[y_col]
            else:
                raise ValueError(f"Метрика {y_col} не найдена в данных")
        else:
            df = data
        
        if df.empty:
            raise ValueError("Данные для построения графика пустые")
        
        # Создаем фигуру
        fig = go.Figure()
        
        # Исходные данные
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df['value'] if 'value' in df.columns else df[y_col],
            mode='lines',
            name='Исходные данные',
            line=dict(color='lightblue', width=1),
            opacity=0.6
        ))
        
        # Сглаженные данные
        if len(df) >= smooth_window:
            y_values = df['value'] if 'value' in df.columns else df[y_col]
            smoothed = y_values.rolling(window=smooth_window, center=True).mean()
            
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=smoothed,
                mode='lines',
                name=f'Скользящее среднее ({smooth_window})',
                line=dict(color='darkblue', width=3)
            ))
        
        # Настройки макета
        fig.update_layout(
            title=title or 'Интерактивная кривая обучения: Вознаграждение',
            xaxis_title='Временные шаги',
            yaxis_title='Вознаграждение за эпизод',
            template=self.theme,
            hovermode='x unified',
            showlegend=True
        )
        
        # Сохранение
        if save_path is None:
            save_path = Path('results/plots/interactive_reward_curve.html')
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        
        logger.info(f"Интерактивный график вознаграждения сохранен: {save_path}")
        return str(save_path)
    
    def plot_interactive_comparison(
        self,
        agents_data: Dict[str, Union[pd.DataFrame, Dict[str, Any]]],
        metric: str = 'episode_reward',
        x_col: str = 'timestep',
        smooth_window: int = 100,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ) -> str:
        """Создать интерактивный сравнительный график агентов.
        
        Args:
            agents_data: Словарь с данными агентов
            metric: Название метрики для сравнения
            x_col: Название колонки для оси X
            smooth_window: Размер окна для сглаживания
            save_path: Путь для сохранения HTML файла
            title: Заголовок графика
            
        Returns:
            Путь к сохраненному HTML файлу
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (agent_name, agent_data) in enumerate(agents_data.items()):
            # Получаем данные для агента
            if isinstance(agent_data, dict):
                if metric in agent_data:
                    df = agent_data[metric]
                else:
                    logger.warning(f"Метрика {metric} не найдена для агента {agent_name}")
                    continue
            else:
                df = agent_data
            
            if df.empty:
                logger.warning(f"Пустые данные для агента {agent_name}")
                continue
            
            color = colors[i % len(colors)]
            y_values = df['value'] if 'value' in df.columns else df[metric]
            
            # Исходные данные
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=y_values,
                mode='lines',
                name=f'{agent_name} (исходные)',
                line=dict(color=color, width=1),
                opacity=0.3,
                showlegend=False
            ))
            
            # Сглаженные данные
            if len(df) >= smooth_window:
                smoothed = pd.Series(y_values).rolling(window=smooth_window, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=smoothed,
                    mode='lines',
                    name=agent_name,
                    line=dict(color=color, width=3)
                ))
        
        # Настройки макета
        fig.update_layout(
            title=title or f'Интерактивное сравнение агентов: {metric.replace("_", " ").title()}',
            xaxis_title='Временные шаги',
            yaxis_title=metric.replace('_', ' ').title(),
            template=self.theme,
            hovermode='x unified',
            showlegend=True
        )
        
        # Сохранение
        if save_path is None:
            save_path = Path(f'results/plots/interactive_comparison_{metric}.html')
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        
        logger.info(f"Интерактивный сравнительный график сохранен: {save_path}")
        return str(save_path)
    
    def create_interactive_dashboard(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ) -> str:
        """Создать интерактивный дашборд с несколькими графиками.
        
        Args:
            data: Данные для построения графиков
            metrics: Список метрик для отображения
            save_path: Путь для сохранения HTML файла
            title: Заголовок дашборда
            
        Returns:
            Путь к сохраненному HTML файлу
        """
        if metrics is None:
            if isinstance(data, dict):
                metrics = list(data.keys())[:4]
            else:
                available_cols = data.columns.tolist()
                metrics = [col for col in ['episode_reward', 'episode_length', 'training_loss'] 
                          if col in available_cols][:4]
        
        n_metrics = len(metrics)
        if n_metrics == 0:
            raise ValueError("Нет доступных метрик для отображения")
        
        # Определяем размер сетки
        if n_metrics <= 2:
            rows, cols = 1, n_metrics
        else:
            rows, cols = 2, 2
        
        # Создаем подграфики
        subplot_titles = [metric.replace('_', ' ').title() for metric in metrics]
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Добавляем графики
        for i, metric in enumerate(metrics):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Получаем данные
            if isinstance(data, dict):
                if metric in data:
                    df = data[metric]
                else:
                    continue
            else:
                df = data
            
            if df.empty:
                continue
            
            x_col = 'timestep' if 'timestep' in df.columns else df.columns[0]
            y_col = 'value' if 'value' in df.columns else metric
            
            if y_col in df.columns:
                # Исходные данные
                fig.add_trace(
                    go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        mode='lines',
                        name=f'{metric} (исходные)',
                        line=dict(width=1),
                        opacity=0.6,
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Сглаженные данные
                if len(df) >= 50:
                    smoothed = pd.Series(df[y_col]).rolling(window=50, center=True).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df[x_col],
                            y=smoothed,
                            mode='lines',
                            name=metric,
                            line=dict(width=3),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
        
        # Настройки макета
        fig.update_layout(
            title_text=title or 'Интерактивный дашборд метрик обучения',
            template=self.theme,
            height=600 if rows == 1 else 800,
            showlegend=False
        )
        
        # Сохранение
        if save_path is None:
            save_path = Path('results/plots/interactive_dashboard.html')
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        
        logger.info(f"Интерактивный дашборд сохранен: {save_path}")
        return str(save_path)


def export_plots_to_formats(
    plot_function: Callable,
    save_dir: Union[str, Path],
    formats: List[str] = ['png', 'pdf', 'svg'],
    **plot_kwargs
) -> Dict[str, str]:
    """Экспортировать график в несколько форматов.
    
    Args:
        plot_function: Функция для создания графика
        save_dir: Директория для сохранения
        formats: Список форматов для экспорта
        **plot_kwargs: Аргументы для функции построения графика
        
    Returns:
        Словарь с путями к сохраненным файлам
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    for fmt in formats:
        # Определяем имя файла
        base_name = plot_kwargs.get('title', 'plot').lower().replace(' ', '_')
        file_path = save_dir / f"{base_name}.{fmt}"
        
        try:
            # Вызываем функцию построения графика
            plot_kwargs['save_path'] = file_path
            result_path = plot_function(**plot_kwargs)
            saved_files[fmt] = result_path
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте в формат {fmt}: {e}")
    
    logger.info(f"График экспортирован в форматы: {list(saved_files.keys())}")
    return saved_files


def create_performance_report(
    data: Union[pd.DataFrame, Dict[str, Any], MetricsTracker],
    output_dir: Union[str, Path] = 'results/performance_report',
    include_interactive: bool = True,
    include_static: bool = True
) -> str:
    """Создать полный отчет о производительности обучения.
    
    Args:
        data: Данные метрик обучения
        output_dir: Директория для сохранения отчета
        include_interactive: Включить интерактивные графики
        include_static: Включить статические графики
        
    Returns:
        Путь к директории с отчетом
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Подготавливаем данные
    if isinstance(data, MetricsTracker):
        data = DataLoader.load_from_metrics_tracker(data)
    
    # Создаем статические графики
    if include_static:
        static_plotter = PerformancePlotter()
        static_dir = output_dir / 'static'
        static_dir.mkdir(exist_ok=True)
        
        try:
            # График вознаграждения
            static_plotter.plot_reward_curve(
                data, 
                save_path=static_dir / 'reward_curve.png',
                title='Кривая обучения: Вознаграждение'
            )
            
            # График длины эпизодов
            if 'episode_length' in data:
                static_plotter.plot_episode_lengths(
                    data,
                    save_path=static_dir / 'episode_lengths.png',
                    title='Длина эпизодов'
                )
            
            # График потерь
            loss_metrics = [k for k in data.keys() if 'loss' in k.lower()]
            if loss_metrics:
                static_plotter.plot_loss_curves(
                    data,
                    loss_columns=loss_metrics,
                    save_path=static_dir / 'loss_curves.png',
                    title='Функции потерь'
                )
            
            # Дашборд
            static_plotter.create_dashboard(
                data,
                save_path=static_dir / 'dashboard.png',
                title='Дашборд метрик обучения'
            )
            
        except Exception as e:
            logger.error(f"Ошибка при создании статических графиков: {e}")
    
    # Создаем интерактивные графики
    if include_interactive:
        interactive_plotter = InteractivePlotter()
        interactive_dir = output_dir / 'interactive'
        interactive_dir.mkdir(exist_ok=True)
        
        try:
            # Интерактивный график вознаграждения
            interactive_plotter.plot_interactive_reward_curve(
                data,
                save_path=interactive_dir / 'reward_curve.html',
                title='Интерактивная кривая обучения'
            )
            
            # Интерактивный дашборд
            interactive_plotter.create_interactive_dashboard(
                data,
                save_path=interactive_dir / 'dashboard.html',
                title='Интерактивный дашборд метрик'
            )
            
        except Exception as e:
            logger.error(f"Ошибка при создании интерактивных графиков: {e}")
    
    logger.info(f"Отчет о производительности создан: {output_dir}")
    return str(output_dir)


# Удобные функции для быстрого использования
def quick_reward_plot(
    data_source: Union[str, Path, pd.DataFrame, Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    interactive: bool = False
) -> str:
    """Быстро создать график вознаграждения.
    
    Args:
        data_source: Источник данных (файл или данные)
        save_path: Путь для сохранения
        interactive: Создать интерактивный график
        
    Returns:
        Путь к сохраненному файлу
    """
    # Загружаем данные
    if isinstance(data_source, (str, Path)):
        data_source = Path(data_source)
        if data_source.suffix == '.csv':
            data = DataLoader.load_from_csv(data_source)
        elif data_source.suffix == '.json':
            data = DataLoader.load_from_json(data_source)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {data_source.suffix}")
    else:
        data = data_source
    
    # Создаем график
    if interactive:
        plotter = InteractivePlotter()
        return plotter.plot_interactive_reward_curve(data, save_path=save_path)
    else:
        plotter = PerformancePlotter()
        return plotter.plot_reward_curve(data, save_path=save_path)


def quick_comparison_plot(
    agents_data: Dict[str, Union[str, Path, pd.DataFrame, Dict[str, Any]]],
    metric: str = 'episode_reward',
    save_path: Optional[Union[str, Path]] = None,
    interactive: bool = False
) -> str:
    """Быстро создать сравнительный график агентов.
    
    Args:
        agents_data: Словарь с данными агентов
        metric: Метрика для сравнения
        save_path: Путь для сохранения
        interactive: Создать интерактивный график
        
    Returns:
        Путь к сохраненному файлу
    """
    # Загружаем данные для каждого агента
    processed_data = {}
    for agent_name, agent_data in agents_data.items():
        if isinstance(agent_data, (str, Path)):
            agent_data = Path(agent_data)
            if agent_data.suffix == '.csv':
                processed_data[agent_name] = DataLoader.load_from_csv(agent_data)
            elif agent_data.suffix == '.json':
                processed_data[agent_name] = DataLoader.load_from_json(agent_data)
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {agent_data.suffix}")
        else:
            processed_data[agent_name] = agent_data
    
    # Создаем график
    if interactive:
        plotter = InteractivePlotter()
        return plotter.plot_interactive_comparison(
            processed_data, metric=metric, save_path=save_path
        )
    else:
        plotter = PerformancePlotter()
        return plotter.plot_multiple_agents(
            processed_data, metric=metric, save_path=save_path
        )