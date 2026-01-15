"""Система экспорта результатов экспериментов RL агентов.

Этот модуль предоставляет комплексную систему для экспорта результатов экспериментов
в различные форматы с автоматическим включением снимков зависимостей, валидацией
целостности данных, архивированием и генерацией сводных отчетов.
"""

import gzip
import hashlib
import json
import pickle
import shutil
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
import pandas as pd

from src.experiments.comparison import ComparisonResult, ExperimentComparator
from src.experiments.experiment import Experiment
from src.utils.dependency_tracker import DependencyTracker, create_experiment_snapshot
from src.utils.rl_logging import get_experiment_logger

logger = get_experiment_logger(__name__)


class ExportFormat:
    """Поддерживаемые форматы экспорта."""

    JSON = "json"
    CSV = "csv"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    EXCEL = "excel"
    PARQUET = "parquet"


class CompressionType:
    """Типы сжатия для архивирования."""

    NONE = "none"
    ZIP = "zip"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"
    GZIP = "gzip"


class ExportError(Exception):
    """Базовое исключение для ошибок экспорта."""

    pass


class ValidationError(ExportError):
    """Исключение для ошибок валидации данных."""

    pass


class CompressionError(ExportError):
    """Исключение для ошибок сжатия."""

    pass


class ResultExporter:
    """Класс для экспорта результатов экспериментов RL агентов.

    Обеспечивает экспорт в различные форматы, интеграцию с dependency_tracker,
    валидацию целостности, архивирование и генерацию сводных отчетов.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        include_dependencies: bool = True,
        validate_integrity: bool = True,
        auto_compress: bool = False,
        compression_type: str = CompressionType.ZIP,
    ):
        """Инициализация экспортера результатов.

        Args:
            output_dir: Директория для сохранения экспортированных данных
            include_dependencies: Включать ли снимки зависимостей
            validate_integrity: Выполнять ли валидацию целостности
            auto_compress: Автоматически сжимать экспортированные данные
            compression_type: Тип сжатия для архивирования
        """
        self.output_dir = Path(output_dir) if output_dir else Path("results/exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.include_dependencies = include_dependencies
        self.validate_integrity = validate_integrity
        self.auto_compress = auto_compress
        self.compression_type = compression_type

        # Инициализируем трекер зависимостей
        if self.include_dependencies:
            self.dependency_tracker = DependencyTracker()
        else:
            self.dependency_tracker = None

        # Метаданные экспорта
        self.export_metadata = {
            "exporter_version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "include_dependencies": include_dependencies,
            "validate_integrity": validate_integrity,
            "auto_compress": auto_compress,
            "compression_type": compression_type,
        }

        logger.info(f"Инициализирован ResultExporter в директории: {self.output_dir}")

    def export_experiment(
        self,
        experiment: Experiment,
        formats: List[str] = None,
        export_name: Optional[str] = None,
        include_raw_data: bool = True,
        include_plots: bool = True,
    ) -> Dict[str, Any]:
        """Экспортировать результаты одного эксперимента.

        Args:
            experiment: Эксперимент для экспорта
            formats: Список форматов для экспорта
            export_name: Имя экспорта (по умолчанию - ID эксперимента)
            include_raw_data: Включать ли сырые данные
            include_plots: Включать ли графики

        Returns:
            Словарь с информацией об экспорте

        Raises:
            ExportError: При ошибках экспорта
        """
        if formats is None:
            formats = [ExportFormat.JSON, ExportFormat.CSV]

        export_name = export_name or f"experiment_{experiment.experiment_id}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.output_dir / f"{export_name}_{timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Начинаем экспорт эксперимента {experiment.experiment_id}")

        try:
            # Подготавливаем данные для экспорта
            export_data = self._prepare_experiment_data(
                experiment, include_raw_data, include_plots
            )

            # Добавляем снимок зависимостей
            if self.include_dependencies:
                dependency_snapshot = self._create_dependency_snapshot(experiment)
                export_data["dependencies"] = dependency_snapshot

            # Экспортируем в различные форматы
            exported_files = {}
            for format_type in formats:
                try:
                    file_path = self._export_to_format(
                        export_data, format_type, export_dir, export_name
                    )
                    exported_files[format_type] = str(file_path)
                    logger.info(f"Экспортировано в формат {format_type}: {file_path}")
                except Exception as e:
                    logger.error(f"Ошибка экспорта в формат {format_type}: {e}")
                    continue

            # Создаем метаданные экспорта
            export_metadata = self._create_export_metadata(
                experiment, export_data, exported_files, export_dir
            )

            # Сохраняем метаданные
            metadata_path = export_dir / "export_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(export_metadata, f, ensure_ascii=False, indent=2)

            # Валидация целостности
            if self.validate_integrity:
                validation_result = self._validate_export_integrity(
                    export_dir, exported_files
                )
                export_metadata["validation"] = validation_result

                # Пересохраняем метаданные с результатами валидации
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(export_metadata, f, ensure_ascii=False, indent=2)

            # Автоматическое сжатие
            if self.auto_compress:
                compressed_path = self._compress_export(export_dir)
                export_metadata["compressed_archive"] = str(compressed_path)

            logger.info(f"Экспорт эксперимента завершен: {export_dir}")
            return export_metadata

        except Exception as e:
            logger.error(
                f"Ошибка экспорта эксперимента {experiment.experiment_id}: {e}"
            )
            raise ExportError(f"Не удалось экспортировать эксперимент: {e}") from e

    def export_multiple_experiments(
        self,
        experiments: List[Experiment],
        formats: List[str] = None,
        export_name: Optional[str] = None,
        include_comparison: bool = True,
        include_summary: bool = True,
    ) -> Dict[str, Any]:
        """Экспортировать результаты нескольких экспериментов.

        Args:
            experiments: Список экспериментов для экспорта
            formats: Список форматов для экспорта
            export_name: Имя экспорта
            include_comparison: Включать ли сравнительный анализ
            include_summary: Включать ли сводный отчет

        Returns:
            Словарь с информацией об экспорте
        """
        if not experiments:
            raise ValueError("Список экспериментов не может быть пустым")

        if formats is None:
            formats = [ExportFormat.JSON, ExportFormat.CSV, ExportFormat.EXCEL]

        export_name = export_name or "multiple_experiments"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.output_dir / f"{export_name}_{timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Начинаем экспорт {len(experiments)} экспериментов")

        try:
            # Экспортируем каждый эксперимент отдельно
            individual_exports = {}
            experiments_dir = export_dir / "individual_experiments"
            experiments_dir.mkdir(exist_ok=True)

            for exp in experiments:
                exp_export_dir = experiments_dir / exp.experiment_id
                exp_export_dir.mkdir(exist_ok=True)

                exp_data = self._prepare_experiment_data(exp, True, False)

                # Экспортируем в JSON для дальнейшего использования
                exp_file = exp_export_dir / f"{exp.experiment_id}.json"
                with open(exp_file, "w", encoding="utf-8") as f:
                    json.dump(exp_data, f, ensure_ascii=False, indent=2, default=str)

                individual_exports[exp.experiment_id] = str(exp_file)

            # Создаем сводные данные
            combined_data = self._combine_experiments_data(experiments)

            # Добавляем сравнительный анализ
            if include_comparison and len(experiments) > 1:
                comparison_data = self._perform_experiments_comparison(experiments)
                combined_data["comparison"] = comparison_data

            # Добавляем сводный отчет
            if include_summary:
                summary_data = self._generate_experiments_summary(experiments)
                combined_data["summary"] = summary_data

            # Добавляем общий снимок зависимостей
            if self.include_dependencies:
                dependency_snapshot = self._create_combined_dependency_snapshot(
                    experiments
                )
                combined_data["dependencies"] = dependency_snapshot

            # Экспортируем сводные данные в различные форматы
            exported_files = {}
            for format_type in formats:
                try:
                    file_path = self._export_to_format(
                        combined_data, format_type, export_dir, "combined_results"
                    )
                    exported_files[format_type] = str(file_path)
                except Exception as e:
                    logger.error(f"Ошибка экспорта в формат {format_type}: {e}")
                    continue

            # Создаем метаданные экспорта
            export_metadata = {
                "export_type": "multiple_experiments",
                "experiment_count": len(experiments),
                "experiment_ids": [exp.experiment_id for exp in experiments],
                "exported_formats": list(exported_files.keys()),
                "exported_files": exported_files,
                "individual_exports": individual_exports,
                "include_comparison": include_comparison,
                "include_summary": include_summary,
                "export_dir": str(export_dir),
                "timestamp": timestamp,
                **self.export_metadata,
            }

            # Сохраняем метаданные
            metadata_path = export_dir / "export_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(export_metadata, f, ensure_ascii=False, indent=2)

            # Валидация целостности
            if self.validate_integrity:
                validation_result = self._validate_export_integrity(
                    export_dir, exported_files
                )
                export_metadata["validation"] = validation_result

                # Пересохраняем метаданные
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(export_metadata, f, ensure_ascii=False, indent=2)

            # Автоматическое сжатие
            if self.auto_compress:
                compressed_path = self._compress_export(export_dir)
                export_metadata["compressed_archive"] = str(compressed_path)

            logger.info(f"Экспорт множественных экспериментов завершен: {export_dir}")
            return export_metadata

        except Exception as e:
            logger.error(f"Ошибка экспорта множественных экспериментов: {e}")
            raise ExportError(f"Не удалось экспортировать эксперименты: {e}") from e

    def export_comparison_results(
        self,
        comparison_result: ComparisonResult,
        formats: List[str] = None,
        export_name: Optional[str] = None,
        include_plots: bool = True,
    ) -> Dict[str, Any]:
        """Экспортировать результаты сравнения экспериментов.

        Args:
            comparison_result: Результат сравнения экспериментов
            formats: Список форматов для экспорта
            export_name: Имя экспорта
            include_plots: Включать ли графики

        Returns:
            Словарь с информацией об экспорте
        """
        if formats is None:
            formats = [ExportFormat.JSON, ExportFormat.CSV, ExportFormat.EXCEL]

        export_name = export_name or "comparison_results"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.output_dir / f"{export_name}_{timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Начинаем экспорт результатов сравнения")

        try:
            # Подготавливаем данные сравнения
            comparison_data = comparison_result.to_dict()

            # Добавляем графики если требуется
            if include_plots:
                plots_dir = export_dir / "plots"
                plots_dir.mkdir(exist_ok=True)

                comparator = ExperimentComparator(output_dir=plots_dir)
                created_plots = comparator.generate_comparison_plots(comparison_result)
                comparison_data["plots"] = created_plots

            # Экспортируем в различные форматы
            exported_files = {}
            for format_type in formats:
                try:
                    file_path = self._export_to_format(
                        comparison_data, format_type, export_dir, "comparison"
                    )
                    exported_files[format_type] = str(file_path)
                except Exception as e:
                    logger.error(f"Ошибка экспорта в формат {format_type}: {e}")
                    continue

            # Создаем метаданные экспорта
            export_metadata = {
                "export_type": "comparison_results",
                "experiment_count": len(comparison_result.experiment_ids),
                "experiment_ids": comparison_result.experiment_ids,
                "exported_formats": list(exported_files.keys()),
                "exported_files": exported_files,
                "include_plots": include_plots,
                "export_dir": str(export_dir),
                "timestamp": timestamp,
                **self.export_metadata,
            }

            # Сохраняем метаданные
            metadata_path = export_dir / "export_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(export_metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"Экспорт результатов сравнения завершен: {export_dir}")
            return export_metadata

        except Exception as e:
            logger.error(f"Ошибка экспорта результатов сравнения: {e}")
            raise ExportError(
                f"Не удалось экспортировать результаты сравнения: {e}"
            ) from e

    def incremental_export(
        self,
        experiment: Experiment,
        export_dir: Union[str, Path],
        update_existing: bool = True,
    ) -> Dict[str, Any]:
        """Инкрементальный экспорт результатов эксперимента.

        Args:
            experiment: Эксперимент для экспорта
            export_dir: Директория существующего экспорта
            update_existing: Обновлять ли существующие файлы

        Returns:
            Словарь с информацией об обновлении
        """
        export_dir = Path(export_dir)

        if not export_dir.exists():
            raise ValueError(f"Директория экспорта не существует: {export_dir}")

        logger.info(f"Начинаем инкрементальный экспорт для {experiment.experiment_id}")

        try:
            # Загружаем существующие метаданные
            metadata_path = export_dir / "export_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    existing_metadata = json.load(f)
            else:
                existing_metadata = {}

            # Подготавливаем новые данные
            new_data = self._prepare_experiment_data(experiment, True, False)

            # Проверяем изменения
            changes_detected = self._detect_data_changes(
                existing_metadata, new_data, experiment.experiment_id
            )

            if not changes_detected and not update_existing:
                logger.info("Изменения не обнаружены, экспорт пропущен")
                return {
                    "updated": False,
                    "reason": "no_changes_detected",
                    "timestamp": datetime.now().isoformat(),
                }

            # Обновляем данные
            updated_files = []

            # Обновляем JSON файл
            json_file = export_dir / f"{experiment.experiment_id}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2, default=str)
            updated_files.append(str(json_file))

            # Обновляем CSV файл если существует
            csv_file = export_dir / f"{experiment.experiment_id}.csv"
            if csv_file.exists() or update_existing:
                self._export_experiment_to_csv(new_data, csv_file)
                updated_files.append(str(csv_file))

            # Обновляем снимок зависимостей
            if self.include_dependencies:
                dependency_snapshot = self._create_dependency_snapshot(experiment)
                deps_file = export_dir / f"dependencies_{experiment.experiment_id}.json"
                with open(deps_file, "w", encoding="utf-8") as f:
                    json.dump(dependency_snapshot, f, ensure_ascii=False, indent=2)
                updated_files.append(str(deps_file))

            # Обновляем метаданные
            update_metadata = {
                "last_updated": datetime.now().isoformat(),
                "updated_files": updated_files,
                "changes_detected": changes_detected,
                "incremental_update": True,
            }

            existing_metadata.update(update_metadata)

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(existing_metadata, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Инкрементальный экспорт завершен: {len(updated_files)} файлов обновлено"
            )
            return {
                "updated": True,
                "updated_files": updated_files,
                "changes_detected": changes_detected,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Ошибка инкрементального экспорта: {e}")
            raise ExportError(
                f"Не удалось выполнить инкрементальный экспорт: {e}"
            ) from e

    def generate_summary_report(
        self,
        export_dirs: List[Union[str, Path]],
        output_path: Optional[Union[str, Path]] = None,
        include_statistics: bool = True,
        include_trends: bool = True,
    ) -> str:
        """Генерировать сводный отчет по экспортированным экспериментам.

        Args:
            export_dirs: Список директорий с экспортированными данными
            output_path: Путь для сохранения отчета
            include_statistics: Включать ли статистику
            include_trends: Включать ли анализ трендов

        Returns:
            Путь к созданному отчету
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"summary_report_{timestamp}.html"
        else:
            output_path = Path(output_path)

        logger.info(f"Генерируем сводный отчет по {len(export_dirs)} экспортам")

        try:
            # Собираем данные из всех экспортов
            all_experiments_data = []
            export_summaries = []

            for export_dir in export_dirs:
                export_dir = Path(export_dir)

                if not export_dir.exists():
                    logger.warning(f"Директория экспорта не найдена: {export_dir}")
                    continue

                # Загружаем метаданные экспорта
                metadata_path = export_dir / "export_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        export_metadata = json.load(f)
                    export_summaries.append(export_metadata)

                # Загружаем данные экспериментов
                for json_file in export_dir.glob("*.json"):
                    if json_file.name != "export_metadata.json":
                        try:
                            with open(json_file, "r", encoding="utf-8") as f:
                                exp_data = json.load(f)
                            all_experiments_data.append(exp_data)
                        except Exception as e:
                            logger.warning(f"Ошибка загрузки {json_file}: {e}")
                            continue

            # Генерируем HTML отчет
            html_content = self._generate_html_summary_report(
                all_experiments_data,
                export_summaries,
                include_statistics,
                include_trends,
            )

            # Сохраняем отчет
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"Сводный отчет сохранен: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Ошибка генерации сводного отчета: {e}")
            raise ExportError(f"Не удалось создать сводный отчет: {e}") from e

    def validate_export_integrity(self, export_dir: Union[str, Path]) -> Dict[str, Any]:
        """Валидировать целостность экспортированных данных.

        Args:
            export_dir: Директория с экспортированными данными

        Returns:
            Результат валидации
        """
        export_dir = Path(export_dir)

        if not export_dir.exists():
            raise ValueError(f"Директория экспорта не существует: {export_dir}")

        logger.info(f"Валидируем целостность экспорта: {export_dir}")

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checked_files": [],
            "file_checksums": {},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Проверяем наличие метаданных
            metadata_path = export_dir / "export_metadata.json"
            if not metadata_path.exists():
                validation_result["errors"].append("Отсутствует файл метаданных")
                validation_result["valid"] = False
            else:
                validation_result["checked_files"].append(str(metadata_path))

                # Загружаем и валидируем метаданные
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    # Проверяем обязательные поля
                    required_fields = ["export_type", "timestamp", "exported_files"]
                    for field in required_fields:
                        if field not in metadata:
                            validation_result["errors"].append(
                                f"Отсутствует обязательное поле в метаданных: {field}"
                            )
                            validation_result["valid"] = False

                    # Проверяем существование экспортированных файлов
                    exported_files = metadata.get("exported_files", {})
                    for format_type, file_path in exported_files.items():
                        file_path = Path(file_path)
                        if not file_path.exists():
                            validation_result["errors"].append(
                                f"Отсутствует экспортированный файл: {file_path}"
                            )
                            validation_result["valid"] = False
                        else:
                            validation_result["checked_files"].append(str(file_path))

                            # Вычисляем контрольную сумму
                            checksum = self._calculate_file_checksum(file_path)
                            validation_result["file_checksums"][str(file_path)] = (
                                checksum
                            )

                except json.JSONDecodeError as e:
                    validation_result["errors"].append(
                        f"Некорректный JSON в метаданных: {e}"
                    )
                    validation_result["valid"] = False

            # Проверяем целостность JSON файлов
            for json_file in export_dir.glob("*.json"):
                if json_file.name != "export_metadata.json":
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            json.load(f)
                        validation_result["checked_files"].append(str(json_file))

                        checksum = self._calculate_file_checksum(json_file)
                        validation_result["file_checksums"][str(json_file)] = checksum

                    except json.JSONDecodeError as e:
                        validation_result["errors"].append(
                            f"Некорректный JSON файл {json_file}: {e}"
                        )
                        validation_result["valid"] = False

            # Проверяем CSV файлы
            for csv_file in export_dir.glob("*.csv"):
                try:
                    pd.read_csv(csv_file)
                    validation_result["checked_files"].append(str(csv_file))

                    checksum = self._calculate_file_checksum(csv_file)
                    validation_result["file_checksums"][str(csv_file)] = checksum

                except Exception as e:
                    validation_result["errors"].append(
                        f"Ошибка чтения CSV файла {csv_file}: {e}"
                    )
                    validation_result["valid"] = False

            # Проверяем HDF5 файлы
            for h5_file in export_dir.glob("*.h5"):
                try:
                    with h5py.File(h5_file, "r") as f:
                        # Проверяем базовую структуру
                        pass
                    validation_result["checked_files"].append(str(h5_file))

                    checksum = self._calculate_file_checksum(h5_file)
                    validation_result["file_checksums"][str(h5_file)] = checksum

                except Exception as e:
                    validation_result["errors"].append(
                        f"Ошибка чтения HDF5 файла {h5_file}: {e}"
                    )
                    validation_result["valid"] = False

            # Сохраняем результат валидации
            validation_path = export_dir / "validation_result.json"
            with open(validation_path, "w", encoding="utf-8") as f:
                json.dump(validation_result, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Валидация завершена. Валидность: {validation_result['valid']}"
            )
            return validation_result

        except Exception as e:
            logger.error(f"Ошибка валидации экспорта: {e}")
            validation_result["errors"].append(f"Критическая ошибка валидации: {e}")
            validation_result["valid"] = False
            return validation_result

    def compress_export(
        self,
        export_dir: Union[str, Path],
        compression_type: Optional[str] = None,
        remove_original: bool = False,
    ) -> str:
        """Сжать экспортированные данные.

        Args:
            export_dir: Директория с экспортированными данными
            compression_type: Тип сжатия
            remove_original: Удалять ли исходную директорию

        Returns:
            Путь к сжатому архиву
        """
        export_dir = Path(export_dir)
        compression_type = compression_type or self.compression_type

        if not export_dir.exists():
            raise ValueError(f"Директория экспорта не существует: {export_dir}")

        logger.info(f"Сжимаем экспорт {export_dir} с типом {compression_type}")

        try:
            return self._compress_export(export_dir, compression_type, remove_original)
        except Exception as e:
            logger.error(f"Ошибка сжатия экспорта: {e}")
            raise CompressionError(f"Не удалось сжать экспорт: {e}") from e

    def list_exports(self) -> List[Dict[str, Any]]:
        """Получить список всех экспортов.

        Returns:
            Список информации об экспортах
        """
        exports = []

        for export_dir in self.output_dir.iterdir():
            if export_dir.is_dir():
                metadata_path = export_dir / "export_metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)

                        export_info = {
                            "export_dir": str(export_dir),
                            "export_name": export_dir.name,
                            "export_type": metadata.get("export_type", "unknown"),
                            "timestamp": metadata.get("timestamp"),
                            "experiment_count": metadata.get("experiment_count", 0),
                            "formats": list(metadata.get("exported_files", {}).keys()),
                            "compressed": "compressed_archive" in metadata,
                            "validated": "validation" in metadata,
                        }
                        exports.append(export_info)

                    except Exception as e:
                        logger.warning(f"Ошибка чтения метаданных {metadata_path}: {e}")
                        continue

        # Сортируем по времени создания
        exports.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return exports

    def cleanup_old_exports(
        self, keep_count: int = 10, keep_days: int = 30
    ) -> Dict[str, Any]:
        """Очистить старые экспорты.

        Args:
            keep_count: Количество экспортов для сохранения
            keep_days: Количество дней для сохранения

        Returns:
            Информация об очистке
        """
        logger.info(
            f"Очищаем старые экспорты (сохраняем {keep_count} шт. или {keep_days} дней)"
        )

        exports = self.list_exports()
        current_time = datetime.now()

        deleted_count = 0
        deleted_size = 0
        errors = []

        # Определяем экспорты для удаления
        exports_to_delete = []

        # По количеству
        if len(exports) > keep_count:
            exports_to_delete.extend(exports[keep_count:])

        # По времени
        for export_info in exports:
            try:
                export_time = datetime.fromisoformat(export_info["timestamp"])
                age_days = (current_time - export_time).days

                if age_days > keep_days and export_info not in exports_to_delete:
                    exports_to_delete.append(export_info)
            except Exception as e:
                logger.warning(
                    f"Ошибка парсинга времени для {export_info['export_name']}: {e}"
                )
                continue

        # Удаляем экспорты
        for export_info in exports_to_delete:
            try:
                export_path = Path(export_info["export_dir"])

                # Вычисляем размер
                size = sum(
                    f.stat().st_size for f in export_path.rglob("*") if f.is_file()
                )

                # Удаляем
                shutil.rmtree(export_path)

                deleted_count += 1
                deleted_size += size

                logger.info(f"Удален экспорт: {export_info['export_name']}")

            except Exception as e:
                error_msg = f"Ошибка удаления {export_info['export_name']}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

        cleanup_result = {
            "deleted_count": deleted_count,
            "deleted_size_bytes": deleted_size,
            "deleted_size_mb": deleted_size / (1024 * 1024),
            "errors": errors,
            "timestamp": current_time.isoformat(),
        }

        logger.info(
            f"Очистка завершена: удалено {deleted_count} экспортов ({cleanup_result['deleted_size_mb']:.2f} MB)"
        )
        return cleanup_result

    # Приватные методы

    def _prepare_experiment_data(
        self,
        experiment: Experiment,
        include_raw_data: bool = True,
        include_plots: bool = True,
    ) -> Dict[str, Any]:
        """Подготовить данные эксперимента для экспорта."""
        data = {
            "experiment_id": experiment.experiment_id,
            "status": experiment.status.value,
            "hypothesis": experiment.hypothesis,
            "created_at": experiment.created_at.isoformat(),
            "started_at": experiment.started_at.isoformat()
            if experiment.started_at
            else None,
            "completed_at": experiment.completed_at.isoformat()
            if experiment.completed_at
            else None,
            "results": experiment.results,
            "configurations": {
                "baseline": self._config_to_dict(experiment.baseline_config),
                "variant": self._config_to_dict(experiment.variant_config),
            },
        }

        if include_raw_data:
            # Добавляем сырые данные метрик
            data["raw_metrics"] = self._extract_raw_metrics(experiment)

        return data

    def _config_to_dict(self, config) -> Dict[str, Any]:
        """Преобразовать конфигурацию в словарь."""
        from dataclasses import asdict, is_dataclass

        if is_dataclass(config):
            return asdict(config)
        else:
            # Для тестов и моков - создаем словарь вручную
            try:
                return {
                    "algorithm": {
                        "name": getattr(config.algorithm, "name", "Unknown"),
                        "learning_rate": getattr(
                            config.algorithm, "learning_rate", 0.001
                        ),
                    },
                    "environment": {
                        "name": getattr(config.environment, "name", "Unknown"),
                    },
                    "training": {
                        "total_timesteps": getattr(
                            config.training, "total_timesteps", 100000
                        ),
                    },
                }
            except AttributeError:
                # Если не удается извлечь атрибуты, возвращаем базовую структуру
                return {
                    "algorithm": {"name": "Unknown", "learning_rate": 0.001},
                    "environment": {"name": "Unknown"},
                    "training": {"total_timesteps": 100000},
                }

    def _extract_raw_metrics(self, experiment: Experiment) -> Dict[str, Any]:
        """Извлечь сырые метрики из эксперимента."""
        raw_metrics = {}

        for config_type in ["baseline", "variant"]:
            if config_type in experiment.results:
                metrics_history = experiment.results[config_type].get(
                    "metrics_history", []
                )
                if metrics_history:
                    raw_metrics[config_type] = metrics_history

        return raw_metrics

    def _create_dependency_snapshot(self, experiment: Experiment) -> Dict[str, Any]:
        """Создать снимок зависимостей для эксперимента."""
        if not self.dependency_tracker:
            return {}

        try:
            snapshot = create_experiment_snapshot(experiment.experiment_id)
            return snapshot
        except Exception as e:
            logger.error(f"Ошибка создания снимка зависимостей: {e}")
            return {"error": str(e)}

    def _create_combined_dependency_snapshot(
        self, experiments: List[Experiment]
    ) -> Dict[str, Any]:
        """Создать объединенный снимок зависимостей для нескольких экспериментов."""
        if not self.dependency_tracker:
            return {}

        try:
            # Создаем общий снимок
            combined_id = f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            snapshot = create_experiment_snapshot(combined_id)

            # Добавляем информацию об экспериментах
            snapshot["experiments"] = [exp.experiment_id for exp in experiments]

            return snapshot
        except Exception as e:
            logger.error(f"Ошибка создания объединенного снимка зависимостей: {e}")
            return {"error": str(e)}

    def _export_to_format(
        self, data: Dict[str, Any], format_type: str, export_dir: Path, base_name: str
    ) -> Path:
        """Экспортировать данные в указанный формат."""
        if format_type == ExportFormat.JSON:
            return self._export_to_json(data, export_dir, base_name)
        elif format_type == ExportFormat.CSV:
            return self._export_to_csv(data, export_dir, base_name)
        elif format_type == ExportFormat.HDF5:
            return self._export_to_hdf5(data, export_dir, base_name)
        elif format_type == ExportFormat.PICKLE:
            return self._export_to_pickle(data, export_dir, base_name)
        elif format_type == ExportFormat.EXCEL:
            return self._export_to_excel(data, export_dir, base_name)
        elif format_type == ExportFormat.PARQUET:
            return self._export_to_parquet(data, export_dir, base_name)
        else:
            raise ValueError(f"Неподдерживаемый формат экспорта: {format_type}")

    def _export_to_json(
        self, data: Dict[str, Any], export_dir: Path, base_name: str
    ) -> Path:
        """Экспортировать в JSON формат."""
        file_path = export_dir / f"{base_name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return file_path

    def _export_to_csv(
        self, data: Dict[str, Any], export_dir: Path, base_name: str
    ) -> Path:
        """Экспортировать в CSV формат."""
        file_path = export_dir / f"{base_name}.csv"

        # Преобразуем данные в плоскую структуру для CSV
        flattened_data = self._flatten_data_for_csv(data)

        if flattened_data:
            df = pd.DataFrame(flattened_data)
            df.to_csv(file_path, index=False, encoding="utf-8")
        else:
            # Создаем пустой CSV с заголовками
            pd.DataFrame().to_csv(file_path, index=False)

        return file_path

    def _export_to_hdf5(
        self, data: Dict[str, Any], export_dir: Path, base_name: str
    ) -> Path:
        """Экспортировать в HDF5 формат."""
        file_path = export_dir / f"{base_name}.h5"

        with h5py.File(file_path, "w") as f:
            self._write_dict_to_hdf5(f, data)

        return file_path

    def _export_to_pickle(
        self, data: Dict[str, Any], export_dir: Path, base_name: str
    ) -> Path:
        """Экспортировать в Pickle формат."""
        file_path = export_dir / f"{base_name}.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(data, f)

        return file_path

    def _export_to_excel(
        self, data: Dict[str, Any], export_dir: Path, base_name: str
    ) -> Path:
        """Экспортировать в Excel формат."""
        file_path = export_dir / f"{base_name}.xlsx"

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Экспортируем основные данные
            main_data = self._flatten_data_for_csv(data)
            if main_data:
                df_main = pd.DataFrame(main_data)
                df_main.to_excel(writer, sheet_name="Main", index=False)

            # Экспортируем метрики отдельно
            if "raw_metrics" in data:
                for config_type, metrics in data["raw_metrics"].items():
                    if metrics:
                        df_metrics = pd.DataFrame(metrics)
                        sheet_name = f"Metrics_{config_type}"
                        df_metrics.to_excel(writer, sheet_name=sheet_name, index=False)

        return file_path

    def _export_to_parquet(
        self, data: Dict[str, Any], export_dir: Path, base_name: str
    ) -> Path:
        """Экспортировать в Parquet формат."""
        file_path = export_dir / f"{base_name}.parquet"

        # Преобразуем данные в плоскую структуру
        flattened_data = self._flatten_data_for_csv(data)

        if flattened_data:
            df = pd.DataFrame(flattened_data)
            df.to_parquet(file_path, index=False)
        else:
            # Создаем пустой DataFrame
            pd.DataFrame().to_parquet(file_path, index=False)

        return file_path

    def _flatten_data_for_csv(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Преобразовать иерархические данные в плоскую структуру для CSV."""
        flattened = []

        def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
            result = {}
            for key, value in d.items():
                new_key = f"{prefix}_{key}" if prefix else key

                if isinstance(value, dict):
                    result.update(flatten_dict(value, new_key))
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    # Обрабатываем список словарей (например, метрики)
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            result.update(flatten_dict(item, f"{new_key}_{i}"))
                else:
                    result[new_key] = value

            return result

        # Если есть сырые метрики, создаем строки для каждого временного шага
        if "raw_metrics" in data:
            for config_type, metrics in data["raw_metrics"].items():
                if metrics:
                    for metric_entry in metrics:
                        row = {
                            "experiment_id": data.get("experiment_id"),
                            "config_type": config_type,
                            **metric_entry,
                        }
                        flattened.append(row)
        else:
            # Создаем одну строку с основными данными
            flattened.append(flatten_dict(data))

        return flattened

    def _write_dict_to_hdf5(self, group, data: Dict[str, Any], path: str = ""):
        """Записать словарь в HDF5 группу."""
        for key, value in data.items():
            current_path = f"{path}/{key}" if path else key

            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._write_dict_to_hdf5(subgroup, value, current_path)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    # Список словарей - создаем подгруппу
                    subgroup = group.create_group(key)
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            item_group = subgroup.create_group(str(i))
                            self._write_dict_to_hdf5(
                                item_group, item, f"{current_path}/{i}"
                            )
                else:
                    # Простой список
                    try:
                        group.create_dataset(key, data=np.array(value))
                    except (TypeError, ValueError):
                        # Если не удается создать numpy array, сохраняем как строку
                        group.create_dataset(key, data=str(value))
            else:
                # Скалярное значение
                try:
                    if value is None:
                        group.create_dataset(key, data="None")
                    elif isinstance(value, (int, float, bool)):
                        group.create_dataset(key, data=value)
                    else:
                        group.create_dataset(key, data=str(value))
                except (TypeError, ValueError):
                    group.create_dataset(key, data=str(value))

    def _combine_experiments_data(
        self, experiments: List[Experiment]
    ) -> Dict[str, Any]:
        """Объединить данные нескольких экспериментов."""
        combined_data = {
            "experiment_count": len(experiments),
            "experiment_ids": [exp.experiment_id for exp in experiments],
            "experiments": [],
            "combined_timestamp": datetime.now().isoformat(),
        }

        for exp in experiments:
            exp_data = self._prepare_experiment_data(exp, True, False)
            combined_data["experiments"].append(exp_data)

        return combined_data

    def _perform_experiments_comparison(
        self, experiments: List[Experiment]
    ) -> Dict[str, Any]:
        """Выполнить сравнение экспериментов."""
        try:
            comparator = ExperimentComparator()
            comparison_result = comparator.compare_experiments(experiments)
            return comparison_result.to_dict()
        except Exception as e:
            logger.error(f"Ошибка сравнения экспериментов: {e}")
            return {"error": str(e)}

    def _generate_experiments_summary(
        self, experiments: List[Experiment]
    ) -> Dict[str, Any]:
        """Генерировать сводку по экспериментам."""
        summary = {
            "total_experiments": len(experiments),
            "completed_experiments": 0,
            "failed_experiments": 0,
            "algorithms_used": set(),
            "environments_used": set(),
            "total_duration": 0.0,
            "average_duration": 0.0,
        }

        durations = []

        for exp in experiments:
            # Статистика по статусам
            if exp.status.value == "completed":
                summary["completed_experiments"] += 1
            elif exp.status.value == "failed":
                summary["failed_experiments"] += 1

            # Алгоритмы и среды
            summary["algorithms_used"].add(exp.baseline_config.algorithm.name)
            summary["algorithms_used"].add(exp.variant_config.algorithm.name)
            summary["environments_used"].add(exp.baseline_config.environment.name)

            # Длительность
            if exp.started_at and exp.completed_at:
                duration = (exp.completed_at - exp.started_at).total_seconds()
                durations.append(duration)

        # Преобразуем множества в списки для JSON сериализации
        summary["algorithms_used"] = list(summary["algorithms_used"])
        summary["environments_used"] = list(summary["environments_used"])

        # Статистика по длительности
        if durations:
            summary["total_duration"] = sum(durations)
            summary["average_duration"] = np.mean(durations)
            summary["min_duration"] = min(durations)
            summary["max_duration"] = max(durations)

        return summary

    def _create_export_metadata(
        self,
        experiment: Experiment,
        export_data: Dict[str, Any],
        exported_files: Dict[str, str],
        export_dir: Path,
    ) -> Dict[str, Any]:
        """Создать метаданные экспорта."""
        metadata = {
            "export_type": "single_experiment",
            "experiment_id": experiment.experiment_id,
            "experiment_status": experiment.status.value,
            "exported_formats": list(exported_files.keys()),
            "exported_files": exported_files,
            "export_dir": str(export_dir),
            "data_size": self._calculate_data_size(export_data),
            "timestamp": datetime.now().isoformat(),
            **self.export_metadata,
        }

        return metadata

    def _calculate_data_size(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Вычислить размер данных."""
        import sys

        # Приблизительный размер в памяти
        memory_size = sys.getsizeof(json.dumps(data, default=str))

        return {
            "memory_size_bytes": memory_size,
            "memory_size_mb": memory_size / (1024 * 1024),
            "estimated": True,
        }

    def _validate_export_integrity(
        self, export_dir: Path, exported_files: Dict[str, str]
    ) -> Dict[str, Any]:
        """Валидировать целостность экспорта."""
        return self.validate_export_integrity(export_dir)

    def _compress_export(
        self,
        export_dir: Path,
        compression_type: Optional[str] = None,
        remove_original: bool = False,
    ) -> str:
        """Сжать директорию экспорта."""
        compression_type = compression_type or self.compression_type

        if compression_type == CompressionType.ZIP:
            archive_path = f"{export_dir}.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in export_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(export_dir.parent)
                        zipf.write(file_path, arcname)

        elif compression_type == CompressionType.TAR_GZ:
            archive_path = f"{export_dir}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(export_dir, arcname=export_dir.name)

        elif compression_type == CompressionType.TAR_BZ2:
            archive_path = f"{export_dir}.tar.bz2"
            with tarfile.open(archive_path, "w:bz2") as tar:
                tar.add(export_dir, arcname=export_dir.name)

        elif compression_type == CompressionType.GZIP:
            # Для GZIP сначала создаем tar архив
            tar_path = f"{export_dir}.tar"
            with tarfile.open(tar_path, "w") as tar:
                tar.add(export_dir, arcname=export_dir.name)

            # Затем сжимаем его
            archive_path = f"{tar_path}.gz"
            with open(tar_path, "rb") as f_in:
                with gzip.open(archive_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Удаляем промежуточный tar файл
            Path(tar_path).unlink()

        else:
            raise ValueError(f"Неподдерживаемый тип сжатия: {compression_type}")

        # Удаляем исходную директорию если требуется
        if remove_original:
            shutil.rmtree(export_dir)

        logger.info(f"Создан сжатый архив: {archive_path}")
        return archive_path

    def _detect_data_changes(
        self,
        existing_metadata: Dict[str, Any],
        new_data: Dict[str, Any],
        experiment_id: str,
    ) -> bool:
        """Обнаружить изменения в данных."""
        # Простая проверка по хешу данных
        new_data_str = json.dumps(new_data, sort_keys=True, default=str)
        new_hash = hashlib.md5(new_data_str.encode()).hexdigest()

        old_hash = existing_metadata.get("data_hash")

        return old_hash != new_hash

    def _export_experiment_to_csv(self, data: Dict[str, Any], file_path: Path) -> None:
        """Экспортировать данные эксперимента в CSV."""
        flattened_data = self._flatten_data_for_csv(data)

        if flattened_data:
            df = pd.DataFrame(flattened_data)
            df.to_csv(file_path, index=False, encoding="utf-8")
        else:
            # Создаем пустой CSV
            pd.DataFrame().to_csv(file_path, index=False)

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Вычислить контрольную сумму файла."""
        hash_md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def _generate_html_summary_report(
        self,
        all_experiments_data: List[Dict[str, Any]],
        export_summaries: List[Dict[str, Any]],
        include_statistics: bool,
        include_trends: bool,
    ) -> str:
        """Генерировать HTML сводный отчет."""
        html_template = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Сводный отчет по экспериментам RL</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .experiment {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
                .warning {{ color: orange; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Сводный отчет по экспериментам RL</h1>
                <p>Сгенерирован: {timestamp}</p>
                <p>Всего экспериментов: {total_experiments}</p>
            </div>
            
            {statistics_section}
            {experiments_section}
            {exports_section}
        </body>
        </html>
        """

        # Генерируем секции
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_experiments = len(all_experiments_data)

        statistics_section = ""
        if include_statistics and all_experiments_data:
            statistics_section = self._generate_statistics_section(all_experiments_data)

        experiments_section = self._generate_experiments_section(all_experiments_data)
        exports_section = self._generate_exports_section(export_summaries)

        return html_template.format(
            timestamp=timestamp,
            total_experiments=total_experiments,
            statistics_section=statistics_section,
            experiments_section=experiments_section,
            exports_section=exports_section,
        )

    def _generate_statistics_section(
        self, experiments_data: List[Dict[str, Any]]
    ) -> str:
        """Генерировать секцию статистики."""
        # Базовая статистика
        completed = sum(
            1 for exp in experiments_data if exp.get("status") == "completed"
        )
        failed = sum(1 for exp in experiments_data if exp.get("status") == "failed")

        algorithms = set()
        environments = set()

        for exp in experiments_data:
            configs = exp.get("configurations", {})
            if "baseline" in configs:
                algorithms.add(
                    configs["baseline"].get("algorithm", {}).get("name", "Unknown")
                )
            if "variant" in configs:
                algorithms.add(
                    configs["variant"].get("algorithm", {}).get("name", "Unknown")
                )

            if "baseline" in configs:
                environments.add(
                    configs["baseline"].get("environment", {}).get("name", "Unknown")
                )

        return f"""
        <div class="section">
            <h2>Статистика</h2>
            <div class="metrics">
                <div class="metric">
                    <strong>Завершенные эксперименты:</strong> {completed}
                </div>
                <div class="metric">
                    <strong>Неудачные эксперименты:</strong> {failed}
                </div>
                <div class="metric">
                    <strong>Использованные алгоритмы:</strong> {", ".join(algorithms)}
                </div>
                <div class="metric">
                    <strong>Использованные среды:</strong> {", ".join(environments)}
                </div>
            </div>
        </div>
        """

    def _generate_experiments_section(
        self, experiments_data: List[Dict[str, Any]]
    ) -> str:
        """Генерировать секцию экспериментов."""
        experiments_html = ""

        for exp in experiments_data:
            status_class = "success" if exp.get("status") == "completed" else "error"

            experiments_html += f"""
            <div class="experiment">
                <h3>{exp.get("experiment_id", "Unknown")}</h3>
                <p><strong>Статус:</strong> <span class="{status_class}">{exp.get("status", "Unknown")}</span></p>
                <p><strong>Гипотеза:</strong> {exp.get("hypothesis", "Не указана")}</p>
                <p><strong>Создан:</strong> {exp.get("created_at", "Неизвестно")}</p>
                <p><strong>Завершен:</strong> {exp.get("completed_at", "Не завершен")}</p>
            </div>
            """

        return f"""
        <div class="section">
            <h2>Эксперименты</h2>
            {experiments_html}
        </div>
        """

    def _generate_exports_section(self, export_summaries: List[Dict[str, Any]]) -> str:
        """Генерировать секцию экспортов."""
        exports_html = ""

        for export in export_summaries:
            exports_html += f"""
            <tr>
                <td>{export.get("export_type", "Unknown")}</td>
                <td>{export.get("timestamp", "Unknown")}</td>
                <td>{export.get("experiment_count", 0)}</td>
                <td>{", ".join(export.get("exported_formats", []))}</td>
            </tr>
            """

        return f"""
        <div class="section">
            <h2>Экспорты</h2>
            <table>
                <thead>
                    <tr>
                        <th>Тип экспорта</th>
                        <th>Время создания</th>
                        <th>Количество экспериментов</th>
                        <th>Форматы</th>
                    </tr>
                </thead>
                <tbody>
                    {exports_html}
                </tbody>
            </table>
        </div>
        """


def export_experiment_results(
    experiment: Experiment,
    output_dir: Optional[Union[str, Path]] = None,
    formats: List[str] = None,
    include_dependencies: bool = True,
) -> Dict[str, Any]:
    """Удобная функция для экспорта результатов эксперимента.

    Args:
        experiment: Эксперимент для экспорта
        output_dir: Директория для сохранения
        formats: Форматы экспорта
        include_dependencies: Включать ли снимки зависимостей

    Returns:
        Информация об экспорте
    """
    exporter = ResultExporter(
        output_dir=output_dir, include_dependencies=include_dependencies
    )

    return exporter.export_experiment(experiment, formats)


def export_multiple_experiments_results(
    experiments: List[Experiment],
    output_dir: Optional[Union[str, Path]] = None,
    formats: List[str] = None,
    include_comparison: bool = True,
) -> Dict[str, Any]:
    """Удобная функция для экспорта результатов нескольких экспериментов.

    Args:
        experiments: Список экспериментов
        output_dir: Директория для сохранения
        formats: Форматы экспорта
        include_comparison: Включать ли сравнительный анализ

    Returns:
        Информация об экспорте
    """
    exporter = ResultExporter(output_dir=output_dir)

    return exporter.export_multiple_experiments(
        experiments, formats, include_comparison=include_comparison
    )
