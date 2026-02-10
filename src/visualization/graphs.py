"""Graph generation utilities for RL experiment visualization.

Provides classes for generating:
- Learning curves (reward vs timesteps)
- Comparison plots (A2C vs PPO)
- Gamma comparison plots (hyperparameter study)
"""
import matplotlib
matplotlib.use('Agg')  # Перед plt!
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from pathlib import Path
from typing import Dict, List, Union
import logging
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LearningCurveGenerator:
    """Generator for training learning curve graphs.

    Attributes:
        width: Graph width in inches
        height: Graph height in inches
        dpi: Resolution
    """

    def __init__(self, width: int = 10, height: int = 6, dpi: int = 150) -> None:
        """Initialize graph generator.

        Args:
            width: Graph width
            height: Graph height
            dpi: Resolution
        """
        self.width = width
        self.height = height
        self.dpi = dpi

    def generate_from_metrics(
        self,
        metrics_csv: Union[str, Path],
        output_path: Union[str, Path],
        title: str = "Learning Curve",
        show_ci: bool = True,
        use_eval_data: bool = True,
    ) -> Figure:
        """Generate learning curve from metrics CSV.

        Args:
            metrics_csv: Path to metrics CSV file
            output_path: Path to save PNG output
            title: Graph title
            show_ci: Show confidence interval
            use_eval_data: If True, use eval_log.csv format (timesteps, mean_reward, std_reward).
                           If False, use metrics.csv format (timesteps, reward_mean, reward_std).

        Returns:
            Matplotlib Figure object
        """
        metrics_csv = Path(metrics_csv)
        
        # Проверка существования файла
        if not metrics_csv.exists():
            raise FileNotFoundError(f"Файл не найден: {metrics_csv}")
        
        logger.info(f"Чтение данных из {metrics_csv}")
        df = pd.read_csv(metrics_csv)
        
        # Проверка на пустой DataFrame
        if df.empty:
            raise ValueError(f"Файл {metrics_csv} пуст или содержит только заголовки")
        
        # Определение столбцов в зависимости от типа данных
        if use_eval_data:
            # eval_log.csv формат: timesteps, mean_reward, std_reward
            required_cols = ['timesteps', 'mean_reward']
            optional_cols = ['std_reward']
            
            # Проверка наличия необходимых столбцов
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Отсутствуют необходимые столбцы для eval_log.csv: {missing_cols}. "
                               f"Найдены столбцы: {list(df.columns)}")
            
            # Переименование столбцов для совместимости
            df = df.rename(columns={'mean_reward': 'reward_mean', 'std_reward': 'reward_std'})
            logger.info("Используется формат eval_log.csv (timesteps, mean_reward, std_reward)")
        else:
            # metrics.csv формат: timesteps, reward_mean, reward_std
            required_cols = ['timesteps', 'reward_mean']
            
            # Проверка наличия необходимых столбцов
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Отсутствуют необходимые столбцы для metrics.csv: {missing_cols}. "
                               f"Найдены столбцы: {list(df.columns)}")
            logger.info("Используется формат metrics.csv (timesteps, reward_mean, reward_std)")
        
        # Конвертация столбцов в числовой тип
        for col in ['timesteps', 'reward_mean', 'reward_std']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Удаление строк с NaN в timesteps или reward_mean
        df = df.dropna(subset=['timesteps', 'reward_mean'])
        
        # Проверка после очистки
        if df.empty:
            raise ValueError(f"После очистки данных в {metrics_csv} не осталось валидных записей")
        
        # Сортировка по timesteps
        df = df.sort_values('timesteps').reset_index(drop=True)
        logger.info(f"Загружено {len(df)} записей, диапазон timesteps: {df['timesteps'].min()} - {df['timesteps'].max()}")
        
        # Заполнение NaN значений для reward_std
        if 'reward_std' in df.columns:
            df['reward_std'] = df['reward_std'].fillna(0)
        else:
            df['reward_std'] = 0
        
        fig, ax = plt.subplots(figsize=(self.width, self.height), dpi=self.dpi)

        # Main line
        ax.plot(
            df["timesteps"],
            df["reward_mean"],
            color="#2E86AB",
            linewidth=2,
            label="Mean Reward",
        )

        # Confidence interval
        if show_ci and "reward_std" in df.columns:
            ax.fill_between(
                df["timesteps"],
                df["reward_mean"] - df["reward_std"],
                df["reward_mean"] + df["reward_std"],
                color="#2E86AB",
                alpha=0.2,
                label="±1 Std Dev",
            )

        # Convergence threshold
        ax.axhline(
            y=200,
            color="#E94F37",
            linestyle="--",
            linewidth=1.5,
            label="Convergence Threshold (200)",
        )

        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Average Reward", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)
        
        logger.info(f"График сохранен: {output_path}")

        return fig


class ComparisonPlotGenerator:
    """Generator for comparing multiple experiments."""

    def __init__(self, width: int = 12, height: int = 7, dpi: int = 150) -> None:
        """Initialize comparison plot generator.

        Args:
            width: Graph width
            height: Graph height
            dpi: Resolution
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        self.colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    def generate(
        self,
        experiment_paths: List[Union[str, Path]],
        labels: List[str],
        output_path: Union[str, Path],
        title: str = "Algorithm Comparison",
        use_eval_data: bool = True,
    ) -> Figure:
        """Generate comparison plot for multiple experiments.

        Args:
            experiment_paths: List of paths to metrics CSV files
            labels: List of labels for each experiment
            output_path: Path to save PNG output
            title: Graph title
            use_eval_data: If True, use eval_log.csv format (timesteps, mean_reward, std_reward).
                           If False, use metrics.csv format (timesteps, reward_mean, reward_std).

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(self.width, self.height), dpi=self.dpi)

        for i, (path, label) in enumerate(zip(experiment_paths, labels)):
            path = Path(path)
            
            # Проверка существования файла
            if not path.exists():
                logger.error(f"Файл не найден: {path}")
                continue
            
            logger.info(f"Чтение данных из {path}")
            df = pd.read_csv(path)
            
            # Проверка на пустой DataFrame
            if df.empty:
                logger.warning(f"Файл {path} пуст или содержит только заголовки")
                continue
            
            # Определение столбцов в зависимости от типа данных
            if use_eval_data:
                # eval_log.csv формат: timesteps, mean_reward, std_reward
                required_cols = ['timesteps', 'mean_reward']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.error(f"Отсутствуют необходимые столбцы для eval_log.csv в {path}: {missing_cols}")
                    continue
                
                # Переименование столбцов для совместимости
                df = df.rename(columns={'mean_reward': 'reward_mean', 'std_reward': 'reward_std'})
            else:
                # metrics.csv формат: timesteps, reward_mean, reward_std
                required_cols = ['timesteps', 'reward_mean']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.error(f"Отсутствуют необходимые столбцы для metrics.csv в {path}: {missing_cols}")
                    continue
            
            # Конвертация столбцов в числовой тип
            for col in ['timesteps', 'reward_mean', 'reward_std']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Удаление строк с NaN в timesteps или reward_mean
            df = df.dropna(subset=['timesteps', 'reward_mean'])
            
            # Проверка после очистки
            if df.empty:
                logger.warning(f"После очистки данных в {path} не осталось валидных записей")
                continue
            
            # Сортировка по timesteps
            df = df.sort_values('timesteps').reset_index(drop=True)
            logger.info(f"Загружено {len(df)} записей из {path}")
            
            # Заполнение NaN значений для reward_std
            if 'reward_std' in df.columns:
                df['reward_std'] = df['reward_std'].fillna(0)
            else:
                df['reward_std'] = 0
            
            ax.plot(
                df["timesteps"],
                df["reward_mean"],
                color=self.colors[i % len(self.colors)],
                linewidth=2,
                label=label,
            )
            ax.fill_between(
                df["timesteps"],
                df["reward_mean"] - df.get("reward_std", 0),
                df["reward_mean"] + df.get("reward_std", 0),
                color=self.colors[i % len(self.colors)],
                alpha=0.15,
            )

        ax.axhline(
            y=200,
            color="#E94F37",
            linestyle="--",
            linewidth=1.5,
            label="Threshold (200)",
        )

        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Average Reward", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)
        
        logger.info(f"График сравнения сохранен: {output_path}")

        return fig


class GammaComparisonPlotGenerator(ComparisonPlotGenerator):
    """Generator specifically for gamma hyperparameter comparison."""

    def generate_gamma_comparison(
        self,
        gamma_results: Dict[float, str],
        output_path: str,
        title: str = "Gamma Hyperparameter Comparison",
        use_eval_data: bool = True,
    ) -> Figure:
        """Generate comparison plot for gamma experiments.

        Args:
            gamma_results: Dict mapping gamma values to metrics CSV paths
            output_path: Path to save PNG output
            title: Graph title
            use_eval_data: If True, use eval_log.csv format.

        Returns:
            Matplotlib Figure object
        """
        labels = [f"γ = {g}" for g in gamma_results.keys()]
        return super().generate(
            experiment_paths=list(gamma_results.values()),
            labels=labels,
            output_path=output_path,
            title=title,
            use_eval_data=use_eval_data,
        )


def main() -> None:
    """CLI entry point for graph generation."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate performance graphs for RL experiments"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment ID (e.g., 'ppo_seed42') or comma-separated list for comparison",
    )
    parser.add_argument(
        "--type",
        choices=["learning_curve", "comparison", "gamma_comparison"],
        default="learning_curve",
        help="Graph type (default: learning_curve)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for PNG file",
    )
    parser.add_argument(
        "--title",
        default="Performance Graph",
        help="Graph title",
    )
    parser.add_argument(
        "--use-training-data",
        action="store_true",
        help="Use training data (metrics.csv) instead of evaluation data (eval_log.csv). "
             "By default, uses eval_log.csv with rewards 200+.",
    )
    args = parser.parse_args()

    # Определение типа данных
    use_eval_data = not args.use_training_data
    data_type = "eval_log.csv" if use_eval_data else "metrics.csv"
    logger.info(f"Используется тип данных: {data_type}")

    # Determine graph type and generate
    if args.type == "learning_curve":
        # Single experiment
        experiment_id = args.experiment
        data_path = Path(f"results/experiments/{experiment_id}/{data_type}")

        if not data_path.exists():
            # Try JSON format
            json_path = Path(f"results/experiments/{experiment_id}/{experiment_id}_metrics.json")
            if json_path.exists():
                logger.info(f"Конвертация JSON в CSV: {json_path}")
                # Convert JSON to CSV
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Определение формата CSV в зависимости от типа данных
                if use_eval_data:
                    # eval_log.csv формат: timesteps, mean_reward, std_reward
                    csv_header = 'timesteps,mean_reward,std_reward\n'
                    with open(data_path, 'w', newline='') as f:
                        f.write(csv_header)
                        for m in data.get('metrics', []):
                            ts = m.get('timestep', 0)
                            reward = m.get('reward', 0)
                            # Для eval_log.csv используем mean_reward
                            f.write(f'{ts},{reward},0\n')
                else:
                    # metrics.csv формат: timesteps,walltime,reward_mean,reward_std,episode_count,fps
                    csv_header = 'timesteps,walltime,reward_mean,reward_std,episode_count,fps\n'
                    with open(data_path, 'w', newline='') as f:
                        f.write(csv_header)
                        for m in data.get('metrics', []):
                            ts = m.get('timestep', 0)
                            reward = m.get('reward', 0)
                            f.write(f'{ts},0,{reward},0,0,0\n')
            else:
                logger.error(f"Файл не найден: {data_path}")
                logger.error(f"JSON файл также не найден: {json_path}")
                sys.exit(1)

        generator = LearningCurveGenerator()
        generator.generate_from_metrics(
            metrics_csv=str(data_path),
            output_path=args.output,
            title=args.title,
            use_eval_data=use_eval_data,
        )
        print(f"Graph saved: {args.output}")

    elif args.type in ("comparison", "gamma_comparison"):
        # Multiple experiments
        experiments = args.experiment.split(",")
        data_paths = [
            Path(f"results/experiments/{exp}/{data_type}")
            for exp in experiments
        ]

        # Check and convert if needed
        for i, (exp, dp) in enumerate(zip(experiments, data_paths)):
            if not dp.exists():
                json_path = Path(f"results/experiments/{exp}/{exp}_metrics.json")
                if json_path.exists():
                    logger.info(f"Конвертация JSON в CSV: {json_path}")
                    # Convert JSON to CSV
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    # Определение формата CSV в зависимости от типа данных
                    if use_eval_data:
                        # eval_log.csv формат: timesteps, mean_reward, std_reward
                        csv_header = 'timesteps,mean_reward,std_reward\n'
                        with open(dp, 'w', newline='') as f:
                            f.write(csv_header)
                            for m in data.get('metrics', []):
                                ts = m.get('timestep', 0)
                                reward = m.get('reward', 0)
                                f.write(f'{ts},{reward},0\n')
                    else:
                        # metrics.csv формат: timesteps,walltime,reward_mean,reward_std,episode_count,fps
                        csv_header = 'timesteps,walltime,reward_mean,reward_std,episode_count,fps\n'
                        with open(dp, 'w', newline='') as f:
                            f.write(csv_header)
                            for m in data.get('metrics', []):
                                ts = m.get('timestep', 0)
                                reward = m.get('reward', 0)
                                f.write(f'{ts},0,{reward},0,0,0\n')
                else:
                    logger.error(f"Файл не найден: {dp}")
                    logger.error(f"JSON файл также не найден: {json_path}")
                    sys.exit(1)

        if args.type == "gamma_comparison":
            # Extract gamma values from experiment names like "gamma_090"
            gamma_map = {}
            for exp in experiments:
                if exp.startswith("gamma_"):
                    gamma_value = float("0." + exp.split("_")[1])
                    gamma_map[gamma_value] = f"results/experiments/{exp}/{data_type}"

            generator = GammaComparisonPlotGenerator()
            generator.generate_gamma_comparison(
                gamma_results=gamma_map,
                output_path=args.output,
                title=args.title,
                use_eval_data=use_eval_data,
            )
        else:
            generator = ComparisonPlotGenerator()
            generator.generate(
                experiment_paths=[str(dp) for dp in data_paths],
                labels=experiments,
                output_path=args.output,
                title=args.title,
                use_eval_data=use_eval_data,
            )
        print(f"Comparison graph saved: {args.output}")


if __name__ == "__main__":
    main()
