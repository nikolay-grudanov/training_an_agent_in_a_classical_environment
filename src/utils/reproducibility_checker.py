"""Система проверки воспроизводимости RL экспериментов.

Этот модуль предоставляет инструменты для проверки воспроизводимости экспериментов
машинного обучения, включая верификацию идентичности результатов при повторных
запусках, статистическую проверку различий, диагностику проблем с детерминированностью
и генерацию отчетов о воспроизводимости.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from scipy import stats

from src.utils.dependency_tracker import DependencyTracker
from src.utils.seeding import SeedManager, set_seed
from src.utils.config import RLConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class StrictnessLevel(Enum):
    """Уровни строгости проверки воспроизводимости."""
    
    MINIMAL = "minimal"      # Только базовые проверки
    STANDARD = "standard"    # Стандартные проверки + статистика
    STRICT = "strict"        # Строгие проверки + детальная диагностика
    PARANOID = "paranoid"    # Максимально строгие проверки


class ReproducibilityIssueType(Enum):
    """Типы проблем с воспроизводимостью."""
    
    SEED_MISMATCH = "seed_mismatch"
    ENVIRONMENT_DIFFERENCE = "environment_difference"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    HARDWARE_DIFFERENCE = "hardware_difference"
    ALGORITHM_NONDETERMINISM = "algorithm_nondeterminism"
    STATISTICAL_DIFFERENCE = "statistical_difference"
    TREND_DEVIATION = "trend_deviation"
    CONFIGURATION_MISMATCH = "configuration_mismatch"


@dataclass
class ReproducibilityIssue:
    """Описание проблемы с воспроизводимостью."""
    
    issue_type: ReproducibilityIssueType
    severity: str  # "critical", "warning", "info"
    description: str
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExperimentRun:
    """Данные одного запуска эксперимента."""
    
    run_id: str
    seed: int
    timestamp: str
    config_hash: str
    environment_hash: str
    results: Dict[str, Any]
    metrics: Dict[str, List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReproducibilityReport:
    """Отчет о воспроизводимости эксперимента."""
    
    experiment_id: str
    timestamp: str
    strictness_level: StrictnessLevel
    is_reproducible: bool
    confidence_score: float  # 0.0 - 1.0
    
    runs: List[ExperimentRun] = field(default_factory=list)
    issues: List[ReproducibilityIssue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # Детальная информация
    seed_analysis: Dict[str, Any] = field(default_factory=dict)
    environment_analysis: Dict[str, Any] = field(default_factory=dict)
    dependency_analysis: Dict[str, Any] = field(default_factory=dict)
    algorithm_analysis: Dict[str, Any] = field(default_factory=dict)


class ReproducibilityChecker:
    """Основной класс для проверки воспроизводимости RL экспериментов."""
    
    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        strictness_level: StrictnessLevel = StrictnessLevel.STANDARD,
        dependency_tracker: Optional[DependencyTracker] = None
    ):
        """Инициализация проверщика воспроизводимости.
        
        Args:
            project_root: Корневая директория проекта
            strictness_level: Уровень строгости проверок
            dependency_tracker: Экземпляр трекера зависимостей
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.strictness_level = strictness_level
        
        # Инициализируем трекер зависимостей
        self.dependency_tracker = dependency_tracker or DependencyTracker(self.project_root)
        
        # Директории для хранения данных
        self.reports_dir = self.project_root / "results" / "reproducibility"
        self.runs_dir = self.reports_dir / "runs"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Менеджер сидов
        self.seed_manager = SeedManager()
        
        # Настройки для различных уровней строгости
        self.strictness_config = self._get_strictness_config()
        
        logger.info(f"Инициализирован ReproducibilityChecker (уровень: {strictness_level.value})")
    
    def _get_strictness_config(self) -> Dict[str, Any]:
        """Получить конфигурацию для текущего уровня строгости."""
        configs = {
            StrictnessLevel.MINIMAL: {
                "check_exact_match": True,
                "check_statistical_equivalence": False,
                "check_trend_consistency": False,
                "check_environment_details": False,
                "statistical_alpha": 0.05,
                "tolerance_rtol": 1e-5,
                "tolerance_atol": 1e-8,
                "min_runs_for_stats": 2,
                "trend_window_size": 10
            },
            StrictnessLevel.STANDARD: {
                "check_exact_match": True,
                "check_statistical_equivalence": True,
                "check_trend_consistency": True,
                "check_environment_details": True,
                "statistical_alpha": 0.05,
                "tolerance_rtol": 1e-7,
                "tolerance_atol": 1e-10,
                "min_runs_for_stats": 3,
                "trend_window_size": 20
            },
            StrictnessLevel.STRICT: {
                "check_exact_match": True,
                "check_statistical_equivalence": True,
                "check_trend_consistency": True,
                "check_environment_details": True,
                "statistical_alpha": 0.01,
                "tolerance_rtol": 1e-10,
                "tolerance_atol": 1e-12,
                "min_runs_for_stats": 5,
                "trend_window_size": 50
            },
            StrictnessLevel.PARANOID: {
                "check_exact_match": True,
                "check_statistical_equivalence": True,
                "check_trend_consistency": True,
                "check_environment_details": True,
                "statistical_alpha": 0.001,
                "tolerance_rtol": 1e-15,
                "tolerance_atol": 1e-15,
                "min_runs_for_stats": 10,
                "trend_window_size": 100
            }
        }
        return configs[self.strictness_level]
    
    def register_experiment_run(
        self,
        experiment_id: str,
        config: RLConfig,
        results: Dict[str, Any],
        metrics: Dict[str, List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Зарегистрировать запуск эксперимента для проверки воспроизводимости.
        
        Args:
            experiment_id: Идентификатор эксперимента
            config: Конфигурация эксперимента
            results: Результаты эксперимента
            metrics: Метрики обучения
            metadata: Дополнительные метаданные
            
        Returns:
            Уникальный идентификатор запуска
        """
        run_id = f"{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Создаем хеши для конфигурации и среды
        config_hash = self._compute_config_hash(config)
        environment_hash = self._compute_environment_hash()
        
        # Создаем объект запуска
        run = ExperimentRun(
            run_id=run_id,
            seed=config.seed,
            timestamp=datetime.now().isoformat(),
            config_hash=config_hash,
            environment_hash=environment_hash,
            results=results,
            metrics=metrics,
            metadata=metadata or {}
        )
        
        # Сохраняем данные запуска
        run_file = self.runs_dir / f"{run_id}.json"
        with open(run_file, 'w', encoding='utf-8') as f:
            json.dump(self._run_to_dict(run), f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Зарегистрирован запуск эксперимента: {run_id}")
        return run_id
    
    def check_reproducibility(
        self,
        experiment_id: str,
        reference_run_id: Optional[str] = None,
        target_runs: Optional[List[str]] = None
    ) -> ReproducibilityReport:
        """Проверить воспроизводимость эксперимента.
        
        Args:
            experiment_id: Идентификатор эксперимента
            reference_run_id: ID эталонного запуска (если None, используется первый)
            target_runs: Список ID запусков для сравнения (если None, все запуски)
            
        Returns:
            Отчет о воспроизводимости
        """
        logger.info(f"Начало проверки воспроизводимости для эксперимента: {experiment_id}")
        
        # Загружаем все запуски эксперимента
        all_runs = self._load_experiment_runs(experiment_id)
        
        if len(all_runs) < 2:
            raise ValueError(f"Недостаточно запусков для проверки воспроизводимости: {len(all_runs)}")
        
        # Определяем эталонный запуск
        if reference_run_id:
            reference_run = next((r for r in all_runs if r.run_id == reference_run_id), None)
            if not reference_run:
                raise ValueError(f"Эталонный запуск не найден: {reference_run_id}")
        else:
            reference_run = all_runs[0]
        
        # Определяем целевые запуски
        if target_runs:
            comparison_runs = [r for r in all_runs if r.run_id in target_runs]
        else:
            comparison_runs = [r for r in all_runs if r.run_id != reference_run.run_id]
        
        # Создаем отчет
        report = ReproducibilityReport(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            strictness_level=self.strictness_level,
            is_reproducible=True,
            confidence_score=1.0,
            runs=[reference_run] + comparison_runs
        )
        
        # Выполняем различные проверки
        self._check_seed_consistency(report, reference_run, comparison_runs)
        self._check_environment_consistency(report, reference_run, comparison_runs)
        self._check_configuration_consistency(report, reference_run, comparison_runs)
        
        if self.strictness_config["check_exact_match"]:
            self._check_exact_results_match(report, reference_run, comparison_runs)
        
        if self.strictness_config["check_statistical_equivalence"]:
            self._check_statistical_equivalence(report, reference_run, comparison_runs)
        
        if self.strictness_config["check_trend_consistency"]:
            self._check_trend_consistency(report, reference_run, comparison_runs)
        
        # Анализируем зависимости
        self._analyze_dependencies(report)
        
        # Вычисляем итоговую оценку
        self._compute_final_assessment(report)
        
        # Генерируем рекомендации
        self._generate_recommendations(report)
        
        # Сохраняем отчет
        self._save_report(report)
        
        logger.info(f"Проверка воспроизводимости завершена. Результат: {'✓' if report.is_reproducible else '✗'}")
        return report
    
    def run_reproducibility_test(
        self,
        test_function: Callable[[int], Dict[str, Any]],
        experiment_id: str,
        seeds: List[int],
        config: Optional[RLConfig] = None
    ) -> ReproducibilityReport:
        """Запустить автоматический тест воспроизводимости.
        
        Args:
            test_function: Функция тестирования, принимающая seed и возвращающая результаты
            experiment_id: Идентификатор эксперимента
            seeds: Список сидов для тестирования
            config: Конфигурация эксперимента
            
        Returns:
            Отчет о воспроизводимости
        """
        logger.info(f"Запуск автоматического теста воспроизводимости: {experiment_id}")
        
        # Создаем конфигурацию по умолчанию если не предоставлена
        if config is None:
            from src.utils.config import RLConfig
            config = RLConfig(experiment_name=experiment_id)
        
        run_ids = []
        
        # Выполняем тесты для каждого сида
        for seed in seeds:
            logger.info(f"Выполнение теста с seed: {seed}")
            
            # Обновляем конфигурацию
            config.seed = seed
            config.apply_seeds()
            
            try:
                # Выполняем тестовую функцию
                results = test_function(seed)
                
                # Извлекаем метрики если они есть
                metrics = results.pop('metrics', {})
                
                # Регистрируем запуск
                run_id = self.register_experiment_run(
                    experiment_id=experiment_id,
                    config=config,
                    results=results,
                    metrics=metrics,
                    metadata={'test_seed': seed}
                )
                run_ids.append(run_id)
                
            except Exception as e:
                logger.error(f"Ошибка при выполнении теста с seed {seed}: {e}")
                # Регистрируем неудачный запуск
                run_id = self.register_experiment_run(
                    experiment_id=experiment_id,
                    config=config,
                    results={'error': str(e)},
                    metrics={},
                    metadata={'test_seed': seed, 'failed': True}
                )
                run_ids.append(run_id)
        
        # Проверяем воспроизводимость
        return self.check_reproducibility(experiment_id)
    
    def validate_determinism(
        self,
        test_function: Callable[[], Any],
        seed: int,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """Проверить детерминированность функции.
        
        Args:
            test_function: Функция для тестирования
            seed: Сид для тестирования
            num_runs: Количество запусков
            
        Returns:
            Результат валидации детерминизма
        """
        logger.info(f"Валидация детерминизма с seed {seed}, {num_runs} запусков")
        
        results = []
        hashes = []
        
        for i in range(num_runs):
            # Устанавливаем сид перед каждым запуском
            set_seed(seed)
            
            try:
                result = test_function()
                results.append(result)
                
                # Вычисляем хеш результата
                result_str = json.dumps(result, sort_keys=True, default=str)
                result_hash = hashlib.sha256(result_str.encode()).hexdigest()
                hashes.append(result_hash)
                
            except Exception as e:
                logger.error(f"Ошибка в запуске {i+1}: {e}")
                results.append({'error': str(e)})
                hashes.append('error')
        
        # Анализируем результаты
        unique_hashes = set(hashes)
        is_deterministic = len(unique_hashes) == 1 and 'error' not in unique_hashes
        
        validation_result = {
            'is_deterministic': is_deterministic,
            'num_runs': num_runs,
            'unique_results': len(unique_hashes),
            'success_rate': sum(1 for h in hashes if h != 'error') / num_runs,
            'results': results,
            'hashes': hashes,
            'analysis': {
                'all_identical': is_deterministic,
                'has_errors': 'error' in hashes,
                'variability_detected': len(unique_hashes) > 1
            }
        }
        
        if is_deterministic:
            logger.info("✓ Функция детерминирована")
        else:
            logger.warning(f"✗ Функция не детерминирована ({len(unique_hashes)} уникальных результатов)")
        
        return validation_result
    
    def diagnose_reproducibility_issues(
        self,
        experiment_id: str,
        deep_analysis: bool = True
    ) -> Dict[str, Any]:
        """Диагностировать проблемы с воспроизводимостью.
        
        Args:
            experiment_id: Идентификатор эксперимента
            deep_analysis: Выполнить глубокий анализ
            
        Returns:
            Результаты диагностики
        """
        logger.info(f"Диагностика проблем воспроизводимости: {experiment_id}")
        
        diagnosis = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'issues_found': [],
            'recommendations': [],
            'system_analysis': {},
            'dependency_analysis': {},
            'seed_analysis': {},
            'algorithm_analysis': {}
        }
        
        # Загружаем запуски эксперимента
        runs = self._load_experiment_runs(experiment_id)
        
        if len(runs) < 2:
            diagnosis['issues_found'].append({
                'type': 'insufficient_data',
                'description': 'Недостаточно запусков для диагностики',
                'severity': 'critical'
            })
            return diagnosis
        
        # Анализируем системную среду
        if deep_analysis:
            diagnosis['system_analysis'] = self._diagnose_system_environment()
            diagnosis['dependency_analysis'] = self._diagnose_dependencies()
        
        # Анализируем сиды
        diagnosis['seed_analysis'] = self._diagnose_seed_issues(runs)
        
        # Анализируем алгоритмические проблемы
        diagnosis['algorithm_analysis'] = self._diagnose_algorithm_issues(runs)
        
        # Генерируем рекомендации на основе найденных проблем
        diagnosis['recommendations'] = self._generate_diagnosis_recommendations(diagnosis)
        
        logger.info(f"Диагностика завершена. Найдено проблем: {len(diagnosis['issues_found'])}")
        return diagnosis
    
    def generate_reproducibility_guide(
        self,
        experiment_id: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """Сгенерировать руководство по обеспечению воспроизводимости.
        
        Args:
            experiment_id: ID эксперимента для анализа (опционально)
            output_path: Путь для сохранения руководства
            
        Returns:
            Текст руководства
        """
        logger.info("Генерация руководства по воспроизводимости")
        
        guide_sections = []
        
        # Заголовок
        guide_sections.append("# Руководство по обеспечению воспроизводимости RL экспериментов")
        guide_sections.append(f"Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        guide_sections.append("")
        
        # Общие принципы
        guide_sections.extend(self._generate_general_principles())
        
        # Настройка сидов
        guide_sections.extend(self._generate_seed_setup_guide())
        
        # Управление зависимостями
        guide_sections.extend(self._generate_dependency_guide())
        
        # Конфигурация алгоритмов
        guide_sections.extend(self._generate_algorithm_config_guide())
        
        # Проверка воспроизводимости
        guide_sections.extend(self._generate_testing_guide())
        
        # Специфичные рекомендации для эксперимента
        if experiment_id:
            guide_sections.extend(self._generate_experiment_specific_guide(experiment_id))
        
        # Чек-лист
        guide_sections.extend(self._generate_checklist())
        
        guide_text = "\n".join(guide_sections)
        
        # Сохраняем руководство если указан путь
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(guide_text)
            logger.info(f"Руководство сохранено: {output_path}")
        
        return guide_text
    
    def _check_seed_consistency(
        self,
        report: ReproducibilityReport,
        reference_run: ExperimentRun,
        comparison_runs: List[ExperimentRun]
    ) -> None:
        """Проверить консистентность сидов."""
        seed_issues = []
        
        # Проверяем, что все запуски используют одинаковые сиды
        reference_seed = reference_run.seed
        
        for run in comparison_runs:
            if run.seed != reference_seed:
                issue = ReproducibilityIssue(
                    issue_type=ReproducibilityIssueType.SEED_MISMATCH,
                    severity="critical",
                    description=f"Несоответствие сидов: эталонный {reference_seed}, текущий {run.seed}",
                    recommendation="Используйте одинаковые сиды для всех запусков",
                    details={'reference_seed': reference_seed, 'run_seed': run.seed, 'run_id': run.run_id}
                )
                seed_issues.append(issue)
                report.issues.append(issue)
        
        # Анализ сидов
        report.seed_analysis = {
            'reference_seed': reference_seed,
            'all_seeds': [reference_run.seed] + [r.seed for r in comparison_runs],
            'seeds_consistent': len(seed_issues) == 0,
            'unique_seeds': len(set([reference_run.seed] + [r.seed for r in comparison_runs])),
            'issues_found': len(seed_issues)
        }
    
    def _check_environment_consistency(
        self,
        report: ReproducibilityReport,
        reference_run: ExperimentRun,
        comparison_runs: List[ExperimentRun]
    ) -> None:
        """Проверить консистентность среды."""
        env_issues = []
        reference_env_hash = reference_run.environment_hash
        
        for run in comparison_runs:
            if run.environment_hash != reference_env_hash:
                issue = ReproducibilityIssue(
                    issue_type=ReproducibilityIssueType.ENVIRONMENT_DIFFERENCE,
                    severity="warning",
                    description=f"Различия в среде выполнения для запуска {run.run_id}",
                    recommendation="Убедитесь в идентичности среды выполнения",
                    details={'reference_hash': reference_env_hash, 'run_hash': run.environment_hash}
                )
                env_issues.append(issue)
                report.issues.append(issue)
        
        report.environment_analysis = {
            'reference_hash': reference_env_hash,
            'environment_consistent': len(env_issues) == 0,
            'issues_found': len(env_issues)
        }
    
    def _check_configuration_consistency(
        self,
        report: ReproducibilityReport,
        reference_run: ExperimentRun,
        comparison_runs: List[ExperimentRun]
    ) -> None:
        """Проверить консистентность конфигурации."""
        config_issues = []
        reference_config_hash = reference_run.config_hash
        
        for run in comparison_runs:
            if run.config_hash != reference_config_hash:
                issue = ReproducibilityIssue(
                    issue_type=ReproducibilityIssueType.CONFIGURATION_MISMATCH,
                    severity="critical",
                    description=f"Различия в конфигурации для запуска {run.run_id}",
                    recommendation="Используйте идентичные конфигурации для всех запусков",
                    details={'reference_hash': reference_config_hash, 'run_hash': run.config_hash}
                )
                config_issues.append(issue)
                report.issues.append(issue)
        
        # Дополнительный анализ конфигурации
        report.algorithm_analysis = {
            'reference_config_hash': reference_config_hash,
            'configurations_consistent': len(config_issues) == 0,
            'issues_found': len(config_issues)
        }
    
    def _check_exact_results_match(
        self,
        report: ReproducibilityReport,
        reference_run: ExperimentRun,
        comparison_runs: List[ExperimentRun]
    ) -> None:
        """Проверить точное совпадение результатов."""
        rtol = self.strictness_config["tolerance_rtol"]
        atol = self.strictness_config["tolerance_atol"]
        
        for run in comparison_runs:
            # Сравниваем численные результаты
            for key, ref_value in reference_run.results.items():
                if key not in run.results:
                    issue = ReproducibilityIssue(
                        issue_type=ReproducibilityIssueType.STATISTICAL_DIFFERENCE,
                        severity="critical",
                        description=f"Отсутствует результат '{key}' в запуске {run.run_id}",
                        recommendation="Убедитесь в полноте сохранения результатов",
                        details={'missing_key': key, 'run_id': run.run_id}
                    )
                    report.issues.append(issue)
                    continue
                
                run_value = run.results[key]
                
                # Проверяем численные значения
                if isinstance(ref_value, (int, float)) and isinstance(run_value, (int, float)):
                    if not np.isclose(ref_value, run_value, rtol=rtol, atol=atol):
                        issue = ReproducibilityIssue(
                            issue_type=ReproducibilityIssueType.STATISTICAL_DIFFERENCE,
                            severity="critical",
                            description=f"Различие в результате '{key}': {ref_value} vs {run_value}",
                            recommendation="Проверьте детерминированность алгоритма и настройки сидов",
                            details={
                                'key': key,
                                'reference_value': ref_value,
                                'run_value': run_value,
                                'difference': abs(ref_value - run_value),
                                'relative_difference': abs(ref_value - run_value) / max(abs(ref_value), 1e-10)
                            }
                        )
                        report.issues.append(issue)
                
                # Проверяем массивы
                elif isinstance(ref_value, (list, np.ndarray)) and isinstance(run_value, (list, np.ndarray)):
                    ref_array = np.asarray(ref_value)
                    run_array = np.asarray(run_value)
                    
                    if ref_array.shape != run_array.shape:
                        issue = ReproducibilityIssue(
                            issue_type=ReproducibilityIssueType.STATISTICAL_DIFFERENCE,
                            severity="critical",
                            description=f"Различие в размерности '{key}': {ref_array.shape} vs {run_array.shape}",
                            recommendation="Проверьте консистентность алгоритма",
                            details={'key': key, 'ref_shape': ref_array.shape, 'run_shape': run_array.shape}
                        )
                        report.issues.append(issue)
                    elif not np.allclose(ref_array, run_array, rtol=rtol, atol=atol):
                        max_diff = np.max(np.abs(ref_array - run_array))
                        issue = ReproducibilityIssue(
                            issue_type=ReproducibilityIssueType.STATISTICAL_DIFFERENCE,
                            severity="warning",
                            description=f"Различие в массиве '{key}', макс. разность: {max_diff}",
                            recommendation="Проверьте детерминированность операций с массивами",
                            details={'key': key, 'max_difference': max_diff}
                        )
                        report.issues.append(issue)
    
    def _check_statistical_equivalence(
        self,
        report: ReproducibilityReport,
        reference_run: ExperimentRun,
        comparison_runs: List[ExperimentRun]
    ) -> None:
        """Проверить статистическую эквивалентность результатов."""
        alpha = self.strictness_config["statistical_alpha"]
        min_runs = self.strictness_config["min_runs_for_stats"]
        
        if len(comparison_runs) < min_runs - 1:
            logger.warning(f"Недостаточно запусков для статистического анализа: {len(comparison_runs) + 1}")
            return
        
        # Собираем метрики для статистического анализа
        all_runs = [reference_run] + comparison_runs
        
        # Анализируем каждую метрику
        for metric_name in reference_run.metrics.keys():
            metric_values = []
            
            for run in all_runs:
                if metric_name in run.metrics:
                    metric_values.append(run.metrics[metric_name])
            
            if len(metric_values) < min_runs:
                continue
            
            # Выполняем статистические тесты
            self._perform_statistical_tests(report, metric_name, metric_values, alpha)
    
    def _perform_statistical_tests(
        self,
        report: ReproducibilityReport,
        metric_name: str,
        metric_values: List[List[float]],
        alpha: float
    ) -> None:
        """Выполнить статистические тесты для метрики."""
        # Тест на нормальность распределения
        if len(metric_values) >= 3:
            # Берем финальные значения каждого запуска
            final_values = [values[-1] if values else 0.0 for values in metric_values]
            
            # Тест Шапиро-Уилка на нормальность
            if len(final_values) >= 3:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(final_values)
                    
                    # Тест на равенство средних (ANOVA или Kruskal-Wallis)
                    if shapiro_p > alpha:  # Нормальное распределение
                        # Используем ANOVA
                        if len(set(final_values)) > 1:  # Есть вариация
                            f_stat, anova_p = stats.f_oneway(*[[v] for v in final_values])
                            test_name = "ANOVA"
                            test_stat = f_stat
                            test_p = anova_p
                        else:
                            test_name = "ANOVA"
                            test_stat = 0.0
                            test_p = 1.0
                    else:
                        # Используем Kruskal-Wallis
                        if len(set(final_values)) > 1:
                            h_stat, kw_p = stats.kruskal(*[[v] for v in final_values])
                            test_name = "Kruskal-Wallis"
                            test_stat = h_stat
                            test_p = kw_p
                        else:
                            test_name = "Kruskal-Wallis"
                            test_stat = 0.0
                            test_p = 1.0
                    
                    # Сохраняем результаты статистики
                    if metric_name not in report.statistics:
                        report.statistics[metric_name] = {}
                    
                    report.statistics[metric_name].update({
                        'shapiro_stat': shapiro_stat,
                        'shapiro_p': shapiro_p,
                        'normality_test_passed': shapiro_p > alpha,
                        'equality_test': test_name,
                        'equality_stat': test_stat,
                        'equality_p': test_p,
                        'means_equal': test_p > alpha,
                        'final_values': final_values,
                        'mean': np.mean(final_values),
                        'std': np.std(final_values),
                        'cv': np.std(final_values) / np.mean(final_values) if np.mean(final_values) != 0 else 0
                    })
                    
                    # Создаем issue если есть статистически значимые различия
                    if test_p <= alpha:
                        issue = ReproducibilityIssue(
                            issue_type=ReproducibilityIssueType.STATISTICAL_DIFFERENCE,
                            severity="warning",
                            description=f"Статистически значимые различия в метрике '{metric_name}' (p={test_p:.4f})",
                            recommendation="Проверьте источники вариативности в алгоритме",
                            details={
                                'metric': metric_name,
                                'test': test_name,
                                'p_value': test_p,
                                'alpha': alpha,
                                'values': final_values
                            }
                        )
                        report.issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"Ошибка при выполнении статистических тестов для {metric_name}: {e}")
    
    def _check_trend_consistency(
        self,
        report: ReproducibilityReport,
        reference_run: ExperimentRun,
        comparison_runs: List[ExperimentRun]
    ) -> None:
        """Проверить консистентность трендов обучения."""
        window_size = self.strictness_config["trend_window_size"]
        
        for metric_name in reference_run.metrics.keys():
            ref_values = reference_run.metrics[metric_name]
            
            if len(ref_values) < window_size:
                continue
            
            # Вычисляем тренд для эталонного запуска
            ref_trend = self._compute_trend(ref_values, window_size)
            
            # Сравниваем с трендами других запусков
            for run in comparison_runs:
                if metric_name not in run.metrics:
                    continue
                
                run_values = run.metrics[metric_name]
                if len(run_values) < window_size:
                    continue
                
                run_trend = self._compute_trend(run_values, window_size)
                
                # Проверяем корреляцию трендов
                correlation = np.corrcoef(ref_trend, run_trend)[0, 1]
                
                if correlation < 0.8:  # Пороговое значение корреляции
                    issue = ReproducibilityIssue(
                        issue_type=ReproducibilityIssueType.TREND_DEVIATION,
                        severity="warning",
                        description=f"Различие в тренде метрики '{metric_name}' (корреляция: {correlation:.3f})",
                        recommendation="Проверьте консистентность процесса обучения",
                        details={
                            'metric': metric_name,
                            'correlation': correlation,
                            'run_id': run.run_id
                        }
                    )
                    report.issues.append(issue)
    
    def _compute_trend(self, values: List[float], window_size: int) -> np.ndarray:
        """Вычислить тренд временного ряда."""
        if len(values) < window_size:
            return np.array([])
        
        # Скользящее среднее
        moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        
        # Вычисляем производную (тренд)
        trend = np.diff(moving_avg)
        
        return trend
    
    def _analyze_dependencies(self, report: ReproducibilityReport) -> None:
        """Анализировать зависимости для отчета."""
        try:
            # Создаем снимок текущих зависимостей
            snapshot = self.dependency_tracker.create_dependency_snapshot("reproducibility_check")
            
            # Проверяем конфликты
            conflicts = self.dependency_tracker.detect_dependency_conflicts()
            
            report.dependency_analysis = {
                'snapshot_created': True,
                'snapshot_hash': snapshot['metadata']['hash'],
                'conflicts_detected': len(conflicts),
                'conflicts': conflicts,
                'ml_libraries': snapshot['ml_libraries'],
                'system_info': {
                    'python_version': snapshot['system']['python']['version'],
                    'platform': snapshot['system']['platform']['system']
                }
            }
            
            # Добавляем issues для конфликтов
            for conflict in conflicts:
                issue = ReproducibilityIssue(
                    issue_type=ReproducibilityIssueType.DEPENDENCY_CONFLICT,
                    severity="warning",
                    description=f"Конфликт зависимостей: {conflict.get('description', 'Неизвестный конфликт')}",
                    recommendation="Устраните конфликты зависимостей для обеспечения воспроизводимости",
                    details=conflict
                )
                report.issues.append(issue)
        
        except Exception as e:
            logger.error(f"Ошибка при анализе зависимостей: {e}")
            report.dependency_analysis = {'error': str(e)}
    
    def _compute_final_assessment(self, report: ReproducibilityReport) -> None:
        """Вычислить итоговую оценку воспроизводимости."""
        # Подсчитываем критические и предупреждающие проблемы
        critical_issues = sum(1 for issue in report.issues if issue.severity == "critical")
        warning_issues = sum(1 for issue in report.issues if issue.severity == "warning")
        
        # Вычисляем оценку уверенности        
        if critical_issues > 0:
            report.is_reproducible = False
            report.confidence_score = max(0.0, 1.0 - (critical_issues * 0.3 + warning_issues * 0.1))
        else:
            report.is_reproducible = warning_issues <= 2  # Допускаем небольшое количество предупреждений
            report.confidence_score = max(0.5, 1.0 - (warning_issues * 0.1))
        
        # Дополнительные факторы
        if report.seed_analysis.get('seeds_consistent', False):
            report.confidence_score = min(1.0, report.confidence_score + 0.1)
        
        if report.environment_analysis.get('environment_consistent', False):
            report.confidence_score = min(1.0, report.confidence_score + 0.1)
        
        # Ограничиваем диапазон
        report.confidence_score = max(0.0, min(1.0, report.confidence_score))
    
    def _generate_recommendations(self, report: ReproducibilityReport) -> None:
        """Сгенерировать рекомендации на основе найденных проблем."""
        recommendations = set()
        
        # Анализируем типы проблем
        issue_types = [issue.issue_type for issue in report.issues]
        
        if ReproducibilityIssueType.SEED_MISMATCH in issue_types:
            recommendations.add("Используйте одинаковые сиды для всех запусков эксперимента")
            recommendations.add("Настройте автоматическую синхронизацию сидов в конфигурации")
        
        if ReproducibilityIssueType.ENVIRONMENT_DIFFERENCE in issue_types:
            recommendations.add("Зафиксируйте версии всех зависимостей в requirements.txt или environment.yml")
            recommendations.add("Используйте контейнеризацию (Docker) для обеспечения идентичности среды")
        
        if ReproducibilityIssueType.DEPENDENCY_CONFLICT in issue_types:
            recommendations.add("Устраните конфликты зависимостей с помощью pip check или conda")
            recommendations.add("Создайте чистую виртуальную среду для экспериментов")
        
        if ReproducibilityIssueType.STATISTICAL_DIFFERENCE in issue_types:
            recommendations.add("Проверьте настройки детерминизма в PyTorch (deterministic=True, benchmark=False)")
            recommendations.add("Отключите стохастические компоненты алгоритма (например, SDE)")
        
        if ReproducibilityIssueType.TREND_DEVIATION in issue_types:
            recommendations.add("Проверьте консистентность инициализации модели")
            recommendations.add("Убедитесь в детерминированности операций обучения")
        
        # Общие рекомендации
        if not report.is_reproducible:
            recommendations.add("Используйте ReproducibilityChecker для регулярной проверки экспериментов")
            recommendations.add("Создайте снимок зависимостей перед началом эксперимента")
        
        # Рекомендации по уровню строгости
        if self.strictness_level == StrictnessLevel.MINIMAL and report.issues:
            recommendations.add("Рассмотрите повышение уровня строгости проверки до STANDARD")
        
        report.recommendations = list(recommendations)
    
    def _save_report(self, report: ReproducibilityReport) -> None:
        """Сохранить отчет о воспроизводимости."""
        report_file = self.reports_dir / f"report_{report.experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Преобразуем отчет в словарь
        report_dict = self._report_to_dict(report)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Отчет о воспроизводимости сохранен: {report_file}")
    
    def _load_experiment_runs(self, experiment_id: str) -> List[ExperimentRun]:
        """Загрузить все запуски эксперимента."""
        runs = []
        
        for run_file in self.runs_dir.glob(f"{experiment_id}_*.json"):
            try:
                with open(run_file, 'r', encoding='utf-8') as f:
                    run_data = json.load(f)
                    run = self._dict_to_run(run_data)
                    runs.append(run)
            except Exception as e:
                logger.error(f"Ошибка загрузки запуска {run_file}: {e}")
        
        return sorted(runs, key=lambda r: r.timestamp)
    
    def _compute_config_hash(self, config: RLConfig) -> str:
        """Вычислить хеш конфигурации."""
        # Преобразуем конфигурацию в словарь
        from dataclasses import asdict
        config_dict = asdict(config)
        
        # Исключаем изменяемые поля
        config_dict.pop('experiment_name', None)
        
        # Сортируем и хешируем
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _compute_environment_hash(self) -> str:
        """Вычислить хеш среды выполнения."""
        try:
            # Получаем информацию о системе
            system_info = self.dependency_tracker.get_system_info()
            ml_libs = self.dependency_tracker.get_ml_library_versions()
            
            # Создаем хеш на основе ключевой информации
            env_data = {
                'python_version': system_info['python']['version'],
                'platform': system_info['platform']['system'],
                'ml_libraries': ml_libs
            }
            
            env_str = json.dumps(env_data, sort_keys=True, default=str)
            return hashlib.sha256(env_str.encode()).hexdigest()
        
        except Exception as e:
            logger.warning(f"Ошибка при вычислении хеша среды: {e}")
            return "unknown"
    
    def _run_to_dict(self, run: ExperimentRun) -> Dict[str, Any]:
        """Преобразовать запуск в словарь."""
        from dataclasses import asdict
        return asdict(run)
    
    def _dict_to_run(self, data: Dict[str, Any]) -> ExperimentRun:
        """Преобразовать словарь в запуск."""
        return ExperimentRun(**data)
    
    def _report_to_dict(self, report: ReproducibilityReport) -> Dict[str, Any]:
        """Преобразовать отчет в словарь."""
        from dataclasses import asdict
        
        report_dict = asdict(report)
        
        # Преобразуем enum'ы в строки
        report_dict['strictness_level'] = report.strictness_level.value
        
        for issue_dict in report_dict['issues']:
            issue_dict['issue_type'] = issue_dict['issue_type'].value
        
        return report_dict
    
    # Методы для диагностики
    def _diagnose_system_environment(self) -> Dict[str, Any]:
        """Диагностировать системную среду."""
        diagnosis = {}
        
        try:
            # Проверяем PyTorch настройки
            import torch
            diagnosis['torch'] = {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'deterministic': torch.backends.cudnn.deterministic,
                'benchmark': torch.backends.cudnn.benchmark
            }
            
            if not torch.backends.cudnn.deterministic:
                diagnosis['issues'] = diagnosis.get('issues', [])
                diagnosis['issues'].append("PyTorch deterministic mode отключен")
            
            if torch.backends.cudnn.benchmark:
                diagnosis['issues'] = diagnosis.get('issues', [])
                diagnosis['issues'].append("PyTorch benchmark mode включен (может нарушить детерминизм)")
        
        except ImportError:
            diagnosis['torch'] = {'error': 'PyTorch не установлен'}
        
        return diagnosis
    
    def _diagnose_dependencies(self) -> Dict[str, Any]:
        """Диагностировать зависимости."""
        try:
            conflicts = self.dependency_tracker.detect_dependency_conflicts()
            return {
                'conflicts_found': len(conflicts),
                'conflicts': conflicts,
                'status': 'ok' if len(conflicts) == 0 else 'issues_detected'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _diagnose_seed_issues(self, runs: List[ExperimentRun]) -> Dict[str, Any]:
        """Диагностировать проблемы с сидами."""
        seeds = [run.seed for run in runs]
        unique_seeds = set(seeds)
        
        diagnosis = {
            'total_runs': len(runs),
            'unique_seeds': len(unique_seeds),
            'seeds': list(unique_seeds),
            'consistent': len(unique_seeds) == 1
        }
        
        if not diagnosis['consistent']:
            diagnosis['issues'] = [f"Используются разные сиды: {unique_seeds}"]
        
        return diagnosis
    
    def _diagnose_algorithm_issues(self, runs: List[ExperimentRun]) -> Dict[str, Any]:
        """Диагностировать алгоритмические проблемы."""
        config_hashes = [run.config_hash for run in runs]
        unique_configs = set(config_hashes)
        
        diagnosis = {
            'total_runs': len(runs),
            'unique_configs': len(unique_configs),
            'configs_consistent': len(unique_configs) == 1
        }
        
        if not diagnosis['configs_consistent']:
            diagnosis['issues'] = ["Используются разные конфигурации алгоритма"]
        
        return diagnosis
    
    def _generate_diagnosis_recommendations(self, diagnosis: Dict[str, Any]) -> List[str]:
        """Сгенерировать рекомендации на основе диагностики."""
        recommendations = []
        
        # Анализируем системные проблемы
        torch_info = diagnosis.get('system_analysis', {}).get('torch', {})
        if not torch_info.get('deterministic', True):
            recommendations.append("Включите детерминистический режим PyTorch: torch.backends.cudnn.deterministic = True")
        
        if torch_info.get('benchmark', False):
            recommendations.append("Отключите benchmark режим PyTorch: torch.backends.cudnn.benchmark = False")
        
        # Анализируем проблемы с зависимостями
        dep_analysis = diagnosis.get('dependency_analysis', {})
        if dep_analysis.get('conflicts_found', 0) > 0:
            recommendations.append("Устраните конфликты зависимостей")
        
        # Анализируем проблемы с сидами
        seed_analysis = diagnosis.get('seed_analysis', {})
        if not seed_analysis.get('consistent', True):
            recommendations.append("Используйте одинаковые сиды для всех запусков")
        
        # Анализируем алгоритмические проблемы
        algo_analysis = diagnosis.get('algorithm_analysis', {})
        if not algo_analysis.get('configs_consistent', True):
            recommendations.append("Используйте идентичные конфигурации алгоритма")
        
        return recommendations
    
    # Методы для генерации руководства
    def _generate_general_principles(self) -> List[str]:
        """Сгенерировать общие принципы воспроизводимости."""
        return [
            "## Общие принципы воспроизводимости",
            "",
            "1. **Детерминизм**: Все случайные процессы должны быть контролируемы через сиды",
            "2. **Изоляция**: Эксперименты должны выполняться в изолированной среде",
            "3. **Документирование**: Все параметры и зависимости должны быть зафиксированы",
            "4. **Валидация**: Регулярная проверка воспроизводимости результатов",
            "5. **Версионирование**: Контроль версий кода и зависимостей",
            ""
        ]
    
    def _generate_seed_setup_guide(self) -> List[str]:
        """Сгенерировать руководство по настройке сидов."""
        return [
            "## Настройка сидов",
            "",
            "### Базовая настройка",
            "```python",
            "from src.utils.seeding import set_seed",
            "",
            "# Установка глобального сида",
            "set_seed(42)",
            "```",
            "",
            "### Проверка воспроизводимости",
            "```python",
            "from src.utils.seeding import verify_reproducibility",
            "",
            "# Проверка детерминизма",
            "is_reproducible = verify_reproducibility(42)",
            "assert is_reproducible, 'Воспроизводимость не обеспечена!'",
            "```",
            "",
            "### Управление сидами в экспериментах",
            "```python",
            "from src.utils.seeding import SeedManager",
            "",
            "seed_manager = SeedManager(base_seed=42)",
            "seed_manager.set_experiment_seed('experiment_1')",
            "```",
            ""
        ]
    
    def _generate_dependency_guide(self) -> List[str]:
        """Сгенерировать руководство по управлению зависимостями."""
        return [
            "## Управление зависимостями",
            "",
            "### Создание снимка зависимостей",
            "```python",
            "from src.utils.dependency_tracker import DependencyTracker",
            "",
            "tracker = DependencyTracker()",
            "snapshot = tracker.create_dependency_snapshot('experiment_baseline')",
            "```",
            "",
            "### Проверка конфликтов",
            "```python",
            "conflicts = tracker.detect_dependency_conflicts()",
            "if conflicts:",
            "    print('Обнаружены конфликты:', conflicts)",
            "```",
            "",
            "### Экспорт зависимостей",
            "```bash",
            "# Pip requirements",
            "pip freeze > requirements.txt",
            "",
            "# Conda environment",
            "conda env export > environment.yml",
            "```",
            ""
        ]
    
    def _generate_algorithm_config_guide(self) -> List[str]:
        """Сгенерировать руководство по конфигурации алгоритмов."""
        return [
            "## Конфигурация алгоритмов",
            "",
            "### PyTorch настройки",
            "```python",
            "import torch",
            "",
            "# Детерминистический режим",
            "torch.backends.cudnn.deterministic = True",
            "torch.backends.cudnn.benchmark = False",
            "",
            "# Отключение многопоточности (если нужно)",
            "torch.set_num_threads(1)",
            "```",
            "",
            "### Stable-Baselines3 настройки",
            "```python",
            "from stable_baselines3 import PPO",
            "",
            "# Детерминистическая политика",
            "model = PPO(",
            "    'MlpPolicy',",
            "    env,",
            "    seed=42,",
            "    device='cpu',  # Фиксированное устройство",
            "    verbose=1",
            ")",
            "```",
            ""
        ]
    
    def _generate_testing_guide(self) -> List[str]:
        """Сгенерировать руководство по тестированию воспроизводимости."""
        return [
            "## Тестирование воспроизводимости",
            "",
            "### Автоматическая проверка",
            "```python",
            "from src.utils.reproducibility_checker import ReproducibilityChecker",
            "",
            "checker = ReproducibilityChecker()",
            "",
            "def test_function(seed):",
            "    # Ваш код эксперимента",
            "    return {'final_reward': reward}",
            "",
            "report = checker.run_reproducibility_test(",
            "    test_function=test_function,",
            "    experiment_id='test_experiment',",
            "    seeds=[42, 42, 42]  # Одинаковые сиды",
            ")",
            "```",
            "",
            "### Проверка детерминизма",
            "```python",
            "validation = checker.validate_determinism(",
            "    test_function=lambda: model.predict(obs),",
            "    seed=42,",
            "    num_runs=5",
            ")",
            "```",
            ""
        ]
    
    def _generate_experiment_specific_guide(self, experiment_id: str) -> List[str]:
        """Сгенерировать специфичные рекомендации для эксперимента."""
        try:
            runs = self._load_experiment_runs(experiment_id)
            if not runs:
                return ["## Специфичные рекомендации", "", "Нет данных для анализа.", ""]
            
            # Анализируем эксперимент
            seeds = [run.seed for run in runs]
            unique_seeds = set(seeds)
            
            recommendations = [
                "## Специфичные рекомендации для эксперимента",
                "",
                f"**Эксперимент ID**: {experiment_id}",
                f"**Количество запусков**: {len(runs)}",
                f"**Уникальные сиды**: {len(unique_seeds)}",
                ""
            ]
            
            if len(unique_seeds) > 1:
                recommendations.extend([
                    "⚠️ **Проблема**: Используются разные сиды",
                    f"Обнаруженные сиды: {list(unique_seeds)}",
                    "Рекомендация: Используйте один сид для всех запусков",
                    ""
                ])
            else:
                recommendations.extend([
                    "✅ **Хорошо**: Все запуски используют одинаковый сид",
                    ""
                ])
            
            return recommendations
        
        except Exception as e:
            return [
                "## Специфичные рекомендации для эксперимента",
                "",
                f"Ошибка анализа: {e}",
                ""
            ]
    
    def _generate_checklist(self) -> List[str]:
        """Сгенерировать чек-лист воспроизводимости."""
        return [
            "## Чек-лист воспроизводимости",
            "",
            "### Перед началом эксперимента",
            "- [ ] Зафиксирован глобальный сид",
            "- [ ] Создан снимок зависимостей",
            "- [ ] Проверены конфликты зависимостей",
            "- [ ] Настроен детерминистический режим PyTorch",
            "- [ ] Зафиксированы версии всех библиотек",
            "",
            "### Во время эксперимента",
            "- [ ] Используются одинаковые сиды для всех запусков",
            "- [ ] Сохраняются все параметры конфигурации",
            "- [ ] Логируются все метрики и результаты",
            "- [ ] Сохраняются промежуточные состояния модели",
            "",
            "### После эксперимента",
            "- [ ] Проверена воспроизводимость результатов",
            "- [ ] Созданы отчеты о воспроизводимости",
            "- [ ] Сохранены все артефакты эксперимента",
            "- [ ] Документированы все найденные проблемы",
            "",
            "### Регулярные проверки",
            "- [ ] Еженедельная проверка воспроизводимости",
            "- [ ] Обновление снимков зависимостей",
            "- [ ] Валидация детерминизма ключевых функций",
            "- [ ] Анализ трендов воспроизводимости",
            ""
        ]


def create_simple_reproducibility_test(
    algorithm_name: str = "PPO",
    env_name: str = "CartPole-v1",
    total_timesteps: int = 1000
) -> Callable[[int], Dict[str, Any]]:
    """Создать простую функцию для тестирования воспроизводимости.
    
    Args:
        algorithm_name: Название алгоритма
        env_name: Название среды
        total_timesteps: Количество шагов обучения
        
    Returns:
        Функция для тестирования воспроизводимости
    """
    def test_function(seed: int) -> Dict[str, Any]:
        """Простой тест обучения агента."""
        import gymnasium as gym
        from stable_baselines3 import PPO
        
        # Устанавливаем сид
        set_seed(seed)
        
        # Создаем среду
        env = gym.make(env_name)
        env.reset(seed=seed)
        
        # Создаем и обучаем модель
        model = PPO("MlpPolicy", env, seed=seed, verbose=0)
        model.learn(total_timesteps=total_timesteps)
        
        # Тестируем модель
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        episode_length = 0
        
        for _ in range(100):  # Максимум 100 шагов
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            episode_length += 1
            
            if terminated or truncated:
                break
        
        env.close()
        
        return {
            'total_reward': total_reward,
            'episode_length': episode_length,
            'final_obs': obs.tolist() if hasattr(obs, 'tolist') else obs,
            'metrics': {
                'rewards': [total_reward],
                'episode_lengths': [episode_length]
            }
        }
    
    return test_function


def quick_reproducibility_check(
    experiment_id: str = "quick_test",
    num_runs: int = 3,
    seed: int = 42
) -> bool:
    """Быстрая проверка воспроизводимости системы.
    
    Args:
        experiment_id: ID эксперимента
        num_runs: Количество запусков
        seed: Сид для тестирования
        
    Returns:
        True если система воспроизводима
    """
    logger.info(f"Быстрая проверка воспроизводимости (seed={seed}, runs={num_runs})")
    
    checker = ReproducibilityChecker(strictness_level=StrictnessLevel.MINIMAL)
    test_function = create_simple_reproducibility_test()
    
    try:
        report = checker.run_reproducibility_test(
            test_function=test_function,
            experiment_id=experiment_id,
            seeds=[seed] * num_runs
        )
        
        is_reproducible = report.is_reproducible
        confidence = report.confidence_score
        
        logger.info(f"Результат проверки: {'✓' if is_reproducible else '✗'} (уверенность: {confidence:.2f})")
        
        if not is_reproducible:
            logger.warning("Обнаружены проблемы с воспроизводимостью:")
            for issue in report.issues[:3]:  # Показываем первые 3 проблемы
                logger.warning(f"  - {issue.description}")
        
        return is_reproducible
    
    except Exception as e:
        logger.error(f"Ошибка при проверке воспроизводимости: {e}")
        return False


def validate_experiment_reproducibility(
    config: RLConfig,
    num_validation_runs: int = 3
) -> bool:
    """Валидировать воспроизводимость конфигурации эксперимента.
    
    Args:
        config: Конфигурация эксперимента
        num_validation_runs: Количество валидационных запусков
        
    Returns:
        True если конфигурация воспроизводима
    """
    logger.info(f"Валидация воспроизводимости конфигурации: {config.experiment_name}")
    
    # Проверяем настройки воспроизводимости в конфигурации
    is_valid, warnings_list = config.validate_reproducibility()
    
    if warnings_list:
        logger.warning(f"Обнаружены предупреждения в конфигурации: {len(warnings_list)}")
        for warning_msg in warnings_list[:3]:
            logger.warning(f"  - {warning_msg}")
    
    # Выполняем практическую проверку
    try:
        checker = ReproducibilityChecker()
        test_function = create_simple_reproducibility_test(
            algorithm_name=config.algorithm.name,
            env_name=config.environment.name,
            total_timesteps=min(config.training.total_timesteps, 5000)  # Ограничиваем для быстроты
        )
        
        report = checker.run_reproducibility_test(
            test_function=test_function,
            experiment_id=f"validation_{config.experiment_name}",
            seeds=[config.seed] * num_validation_runs,
            config=config
        )
        
        practical_reproducible = report.is_reproducible
        
        logger.info(f"Практическая проверка: {'✓' if practical_reproducible else '✗'}")
        
        return is_valid and practical_reproducible
    
    except Exception as e:
        logger.error(f"Ошибка при практической валидации: {e}")
        return False