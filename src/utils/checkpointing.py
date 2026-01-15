"""Утилиты для создания и восстановления чекпоинтов в RL экспериментах.

Этот модуль предоставляет функции для сохранения и загрузки чекпоинтов моделей
с метаданными, автоматического управления чекпоинтами, валидации целостности
и поддержки восстановления состояния эксперимента.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Метаданные чекпоинта."""

    experiment_id: str
    timestamp: str
    timestep: int
    episode: int
    reward: float
    model_class: str
    algorithm: str
    environment: str
    seed: int
    hyperparameters: Dict[str, Any]
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    python_version: Optional[str] = None
    torch_version: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Создать метаданные из словаря.

        Args:
            data: Словарь с данными

        Returns:
            Объект метаданных
        """
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать метаданные в словарь.

        Returns:
            Словарь с метаданными
        """
        return asdict(self)


class CheckpointManager:
    """Менеджер для управления чекпоинтами моделей."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        experiment_id: str,
        max_checkpoints: int = 5,
        save_best_only: bool = False,
    ):
        """Инициализация менеджера чекпоинтов.

        Args:
            checkpoint_dir: Директория для сохранения чекпоинтов
            experiment_id: Идентификатор эксперимента
            max_checkpoints: Максимальное количество чекпоинтов для хранения
            save_best_only: Сохранять только лучшие чекпоинты
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_id = experiment_id
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only

        # Создаем директорию для чекпоинтов
        self.experiment_dir = self.checkpoint_dir / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Файл с индексом чекпоинтов
        self.index_file = self.experiment_dir / "checkpoints_index.json"

        # Загружаем существующий индекс
        self.checkpoints_index = self._load_index()

        logger.info(
            f"Инициализирован CheckpointManager для эксперимента {experiment_id}"
        )
        logger.info(f"Директория чекпоинтов: {self.experiment_dir}")

    def save_checkpoint(
        self,
        model: Any,
        metadata: CheckpointMetadata,
        optimizer_state: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Сохранить чекпоинт модели.

        Args:
            model: Модель для сохранения
            metadata: Метаданные чекпоинта
            optimizer_state: Состояние оптимизатора (опционально)
            additional_data: Дополнительные данные для сохранения

        Returns:
            Путь к сохраненному чекпоинту

        Raises:
            ValueError: Если модель не поддерживает сохранение
            IOError: Если не удалось сохранить файл
        """
        # Генерируем имя файла чекпоинта
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"checkpoint_{metadata.timestep}_{timestamp_str}.pt"
        checkpoint_path = self.experiment_dir / checkpoint_filename

        try:
            # Подготавливаем данные для сохранения
            checkpoint_data = {
                "metadata": metadata.to_dict(),
                "model_state_dict": self._get_model_state_dict(model),
                "optimizer_state_dict": optimizer_state,
                "additional_data": additional_data or {},
                "save_timestamp": time.time(),
            }

            # Сохраняем чекпоинт
            torch.save(checkpoint_data, checkpoint_path)

            # Вычисляем хеш и размер файла
            file_hash = self._calculate_file_hash(checkpoint_path)
            file_size = checkpoint_path.stat().st_size

            # Обновляем метаданные
            metadata.file_hash = file_hash
            metadata.file_size = file_size
            metadata.python_version = self._get_python_version()
            metadata.torch_version = torch.__version__

            # Добавляем в индекс
            checkpoint_info = {
                "filename": checkpoint_filename,
                "path": str(checkpoint_path),
                "metadata": metadata.to_dict(),
                "created_at": datetime.now().isoformat(),
            }

            self.checkpoints_index.append(checkpoint_info)

            # Управляем количеством чекпоинтов
            self._manage_checkpoints()

            # Сохраняем обновленный индекс
            self._save_index()

            logger.info(f"Чекпоинт сохранен: {checkpoint_path}")
            logger.info(f"Timestep: {metadata.timestep}, Reward: {metadata.reward:.4f}")

            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Ошибка при сохранении чекпоинта: {e}")
            # Удаляем частично созданный файл
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            raise IOError(f"Не удалось сохранить чекпоинт: {e}") from e

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        timestep: Optional[int] = None,
        load_best: bool = False,
    ) -> tuple[Dict[str, Any], CheckpointMetadata]:
        """Загрузить чекпоинт.

        Args:
            checkpoint_path: Путь к конкретному чекпоинту
            timestep: Загрузить чекпоинт для конкретного timestep
            load_best: Загрузить лучший чекпоинт по reward

        Returns:
            Кортеж (данные_чекпоинта, метаданные)

        Raises:
            FileNotFoundError: Если чекпоинт не найден
            ValueError: Если чекпоинт поврежден
        """
        # Определяем какой чекпоинт загружать
        if checkpoint_path:
            target_path = Path(checkpoint_path)
        elif load_best:
            target_path = self._get_best_checkpoint_path()
        elif timestep is not None:
            target_path = self._get_checkpoint_by_timestep(timestep)
        else:
            target_path = self._get_latest_checkpoint_path()

        if not target_path or not target_path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {target_path}")

        try:
            # Загружаем данные
            checkpoint_data = torch.load(target_path, map_location="cpu")

            # Проверяем целостность
            if not self._validate_checkpoint(target_path, checkpoint_data):
                raise ValueError(f"Чекпоинт поврежден: {target_path}")

            # Извлекаем метаданные
            metadata = CheckpointMetadata.from_dict(checkpoint_data["metadata"])

            logger.info(f"Чекпоинт загружен: {target_path}")
            logger.info(f"Timestep: {metadata.timestep}, Reward: {metadata.reward:.4f}")

            return checkpoint_data, metadata

        except Exception as e:
            logger.error(f"Ошибка при загрузке чекпоинта {target_path}: {e}")
            raise ValueError(f"Не удалось загрузить чекпоинт: {e}") from e

    def list_checkpoints(self, sort_by: str = "timestep") -> List[Dict[str, Any]]:
        """Получить список всех чекпоинтов.

        Args:
            sort_by: Поле для сортировки ('timestep', 'reward', 'created_at')

        Returns:
            Список информации о чекпоинтах
        """
        checkpoints = self.checkpoints_index.copy()

        if sort_by == "timestep":
            checkpoints.sort(key=lambda x: x["metadata"]["timestep"])
        elif sort_by == "reward":
            checkpoints.sort(key=lambda x: x["metadata"]["reward"], reverse=True)
        elif sort_by == "created_at":
            checkpoints.sort(key=lambda x: x["created_at"])

        return checkpoints

    def delete_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """Удалить конкретный чекпоинт.

        Args:
            checkpoint_path: Путь к чекпоинту

        Returns:
            True если удаление успешно
        """
        checkpoint_path = Path(checkpoint_path)

        try:
            # Удаляем файл
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            # Удаляем из индекса
            self.checkpoints_index = [
                cp
                for cp in self.checkpoints_index
                if Path(cp["path"]) != checkpoint_path
            ]

            self._save_index()

            logger.info(f"Чекпоинт удален: {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при удалении чекпоинта {checkpoint_path}: {e}")
            return False

    def cleanup_old_checkpoints(self, days_to_keep: int = 7) -> int:
        """Очистить старые чекпоинты.

        Args:
            days_to_keep: Количество дней для хранения

        Returns:
            Количество удаленных чекпоинтов
        """
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        deleted_count = 0

        checkpoints_to_remove = []
        for checkpoint in self.checkpoints_index:
            checkpoint_path = Path(checkpoint["path"])
            if checkpoint_path.exists():
                if checkpoint_path.stat().st_mtime < cutoff_time:
                    if self.delete_checkpoint(checkpoint_path):
                        checkpoints_to_remove.append(checkpoint)
                        deleted_count += 1

        logger.info(f"Удалено {deleted_count} старых чекпоинтов")
        return deleted_count

    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Получить статистику по чекпоинтам.

        Returns:
            Словарь со статистикой
        """
        if not self.checkpoints_index:
            return {
                "total_checkpoints": 0,
                "total_size_mb": 0,
                "best_reward": None,
                "latest_timestep": None,
            }

        total_size = sum(
            cp["metadata"]["file_size"] or 0 for cp in self.checkpoints_index
        )

        rewards = [cp["metadata"]["reward"] for cp in self.checkpoints_index]
        timesteps = [cp["metadata"]["timestep"] for cp in self.checkpoints_index]

        return {
            "total_checkpoints": len(self.checkpoints_index),
            "total_size_mb": total_size / (1024 * 1024),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "avg_reward": sum(rewards) / len(rewards),
            "latest_timestep": max(timesteps),
            "earliest_timestep": min(timesteps),
        }

    def _load_index(self) -> List[Dict[str, Any]]:
        """Загрузить индекс чекпоинтов."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить индекс чекпоинтов: {e}")

        return []

    def _save_index(self) -> None:
        """Сохранить индекс чекпоинтов."""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.checkpoints_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Не удалось сохранить индекс чекпоинтов: {e}")

    def _manage_checkpoints(self) -> None:
        """Управление количеством чекпоинтов."""
        if len(self.checkpoints_index) <= self.max_checkpoints:
            return

        if self.save_best_only:
            # Оставляем только лучшие по reward
            self.checkpoints_index.sort(
                key=lambda x: x["metadata"]["reward"], reverse=True
            )
            checkpoints_to_remove = self.checkpoints_index[self.max_checkpoints :]
        else:
            # Удаляем самые старые
            self.checkpoints_index.sort(key=lambda x: x["metadata"]["timestep"])
            checkpoints_to_remove = self.checkpoints_index[: -self.max_checkpoints]

        # Удаляем файлы
        for checkpoint in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint["path"])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.debug(f"Удален старый чекпоинт: {checkpoint_path}")

        # Обновляем индекс
        if self.save_best_only:
            self.checkpoints_index = self.checkpoints_index[: self.max_checkpoints]
        else:
            self.checkpoints_index = self.checkpoints_index[-self.max_checkpoints :]

    def _get_model_state_dict(self, model: Any) -> Dict[str, Any]:
        """Получить state_dict модели."""
        if hasattr(model, "state_dict"):
            return model.state_dict()
        elif hasattr(model, "save"):
            # Для Stable-Baselines3 моделей
            temp_path = self.experiment_dir / "temp_model.zip"
            model.save(temp_path)
            with open(temp_path, "rb") as f:
                model_data = f.read()
            temp_path.unlink()
            return {"model_data": model_data}
        else:
            raise ValueError("Модель не поддерживает сохранение state_dict")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Вычислить хеш файла."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _validate_checkpoint(
        self, checkpoint_path: Path, checkpoint_data: Dict[str, Any]
    ) -> bool:
        """Проверить целостность чекпоинта."""
        try:
            # Проверяем наличие обязательных полей
            required_fields = ["metadata", "model_state_dict"]
            for field in required_fields:
                if field not in checkpoint_data:
                    logger.error(f"Отсутствует поле {field} в чекпоинте")
                    return False

            # Проверяем хеш файла если есть
            metadata = checkpoint_data["metadata"]
            if "file_hash" in metadata and metadata["file_hash"]:
                current_hash = self._calculate_file_hash(checkpoint_path)
                if current_hash != metadata["file_hash"]:
                    logger.error("Хеш файла не совпадает с ожидаемым")
                    return False

            return True

        except Exception as e:
            logger.error(f"Ошибка при валидации чекпоинта: {e}")
            return False

    def _get_best_checkpoint_path(self) -> Optional[Path]:
        """Получить путь к лучшему чекпоинту."""
        if not self.checkpoints_index:
            return None

        best_checkpoint = max(
            self.checkpoints_index, key=lambda x: x["metadata"]["reward"]
        )
        return Path(best_checkpoint["path"])

    def _get_latest_checkpoint_path(self) -> Optional[Path]:
        """Получить путь к последнему чекпоинту."""
        if not self.checkpoints_index:
            return None

        latest_checkpoint = max(
            self.checkpoints_index, key=lambda x: x["metadata"]["timestep"]
        )
        return Path(latest_checkpoint["path"])

    def _get_checkpoint_by_timestep(self, timestep: int) -> Optional[Path]:
        """Получить чекпоинт для конкретного timestep."""
        for checkpoint in self.checkpoints_index:
            if checkpoint["metadata"]["timestep"] == timestep:
                return Path(checkpoint["path"])
        return None

    def _get_python_version(self) -> str:
        """Получить версию Python."""
        import sys

        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def create_checkpoint_metadata(
    experiment_id: str,
    timestep: int,
    episode: int,
    reward: float,
    model_class: str,
    algorithm: str,
    environment: str,
    seed: int,
    hyperparameters: Dict[str, Any],
) -> CheckpointMetadata:
    """Создать метаданные чекпоинта.

    Args:
        experiment_id: Идентификатор эксперимента
        timestep: Текущий timestep
        episode: Номер эпизода
        reward: Текущая награда
        model_class: Класс модели
        algorithm: Используемый алгоритм
        environment: Среда обучения
        seed: Используемый seed
        hyperparameters: Гиперпараметры

    Returns:
        Объект метаданных чекпоинта
    """
    return CheckpointMetadata(
        experiment_id=experiment_id,
        timestamp=datetime.now().isoformat(),
        timestep=timestep,
        episode=episode,
        reward=reward,
        model_class=model_class,
        algorithm=algorithm,
        environment=environment,
        seed=seed,
        hyperparameters=hyperparameters,
    )


def restore_training_state(
    checkpoint_path: Union[str, Path], model: Any, optimizer: Optional[Any] = None
) -> tuple[CheckpointMetadata, Dict[str, Any]]:
    """Восстановить состояние обучения из чекпоинта.

    Args:
        checkpoint_path: Путь к чекпоинту
        model: Модель для загрузки состояния
        optimizer: Оптимизатор для загрузки состояния (опционально)

    Returns:
        Кортеж (метаданные, дополнительные_данные)

    Raises:
        FileNotFoundError: Если чекпоинт не найден
        ValueError: Если чекпоинт поврежден
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    try:
        # Загружаем чекпоинт
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        # Восстанавливаем состояние модели
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(checkpoint_data["model_state_dict"])
        elif hasattr(model, "load"):
            # Для Stable-Baselines3 моделей
            if "model_data" in checkpoint_data["model_state_dict"]:
                temp_path = checkpoint_path.parent / "temp_restore_model.zip"
                with open(temp_path, "wb") as f:
                    f.write(checkpoint_data["model_state_dict"]["model_data"])
                model.load(temp_path)
                temp_path.unlink()

        # Восстанавливаем состояние оптимизатора
        if optimizer and checkpoint_data.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        # Извлекаем метаданные
        metadata = CheckpointMetadata.from_dict(checkpoint_data["metadata"])
        additional_data = checkpoint_data.get("additional_data", {})

        logger.info(f"Состояние обучения восстановлено из {checkpoint_path}")
        logger.info(f"Timestep: {metadata.timestep}, Episode: {metadata.episode}")

        return metadata, additional_data

    except Exception as e:
        logger.error(f"Ошибка при восстановлении состояния: {e}")
        raise ValueError(f"Не удалось восстановить состояние: {e}") from e
