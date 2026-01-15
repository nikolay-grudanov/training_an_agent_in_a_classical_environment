"""Система отслеживания зависимостей для RL экспериментов.

Этот модуль предоставляет инструменты для отслеживания зависимостей проекта,
создания снимков среды, детектирования изменений и обеспечения воспроизводимости
экспериментов машинного обучения.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil
import yaml

from src.utils.rl_logging import get_logger

logger = get_logger(__name__)


class DependencyTracker:
    """Трекер зависимостей для обеспечения воспроизводимости RL экспериментов."""
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """Инициализация трекера зависимостей.
        
        Args:
            project_root: Корневая директория проекта
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.snapshots_dir = self.project_root / "results" / "dependencies"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Ключевые ML библиотеки для отслеживания
        self.ml_libraries = {
            'torch', 'torchvision', 'torchaudio',
            'stable-baselines3', 'sb3-contrib',
            'gymnasium', 'gym',
            'numpy', 'scipy', 'pandas',
            'matplotlib', 'seaborn', 'plotly',
            'scikit-learn', 'opencv-python',
            'tensorboard', 'wandb',
            'jupyter', 'ipython'
        }
        
        logger.info(f"Инициализирован DependencyTracker для проекта: {self.project_root}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Получить информацию о системе.
        
        Returns:
            Словарь с информацией о системе
        """
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'python': {
                'version': sys.version,
                'version_info': {
                    'major': sys.version_info.major,
                    'minor': sys.version_info.minor,
                    'micro': sys.version_info.micro
                },
                'implementation': platform.python_implementation(),
                'executable': sys.executable,
                'prefix': sys.prefix
            },
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'architecture': platform.architecture()
            },
            'hardware': {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'disk_usage': {
                    'total': psutil.disk_usage('/').total,
                    'free': psutil.disk_usage('/').free
                }
            }
        }
        
        # Информация о GPU (если доступно)
        try:
            import torch
            if torch.cuda.is_available():
                system_info['gpu'] = {
                    'cuda_available': True,
                    'cuda_version': torch.version.cuda,
                    'device_count': torch.cuda.device_count(),
                    'devices': []
                }
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    system_info['gpu']['devices'].append({
                        'name': props.name,
                        'memory_total': props.total_memory,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
            else:
                system_info['gpu'] = {'cuda_available': False}
        except ImportError:
            system_info['gpu'] = {'cuda_available': False, 'torch_not_installed': True}
        
        return system_info
    
    def get_package_manager_info(self) -> Dict[str, Any]:
        """Определить доступные менеджеры пакетов.
        
        Returns:
            Информация о менеджерах пакетов
        """
        managers = {}
        
        # Проверяем pip
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', '--version'],
                capture_output=True, text=True, check=True
            )
            managers['pip'] = {
                'available': True,
                'version': result.stdout.strip()
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            managers['pip'] = {'available': False}
        
        # Проверяем conda
        try:
            result = subprocess.run(
                ['conda', '--version'],
                capture_output=True, text=True, check=True
            )
            managers['conda'] = {
                'available': True,
                'version': result.stdout.strip()
            }
            
            # Получаем информацию о текущей среде
            try:
                env_result = subprocess.run(
                    ['conda', 'info', '--json'],
                    capture_output=True, text=True, check=True
                )
                env_info = json.loads(env_result.stdout)
                managers['conda']['active_env'] = env_info.get('active_prefix_name', 'base')
                managers['conda']['env_path'] = env_info.get('active_prefix', '')
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                pass
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            managers['conda'] = {'available': False}
        
        # Проверяем poetry
        try:
            result = subprocess.run(
                ['poetry', '--version'],
                capture_output=True, text=True, check=True
            )
            managers['poetry'] = {
                'available': True,
                'version': result.stdout.strip()
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            managers['poetry'] = {'available': False}
        
        return managers
    
    def get_pip_packages(self) -> Dict[str, str]:
        """Получить список установленных pip пакетов.
        
        Returns:
            Словарь {package_name: version}
        """
        packages = {}
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'freeze'],
                capture_output=True, text=True, check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if line and '==' in line:
                    name, version = line.split('==', 1)
                    packages[name.lower()] = version
                    
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка получения pip пакетов: {e}")
        
        return packages
    
    def get_conda_packages(self) -> Dict[str, str]:
        """Получить список установленных conda пакетов.
        
        Returns:
            Словарь {package_name: version}
        """
        packages = {}
        
        try:
            result = subprocess.run(
                ['conda', 'list', '--json'],
                capture_output=True, text=True, check=True
            )
            
            conda_packages = json.loads(result.stdout)
            for pkg in conda_packages:
                packages[pkg['name'].lower()] = pkg['version']
                
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            logger.warning("Не удалось получить список conda пакетов")
        
        return packages
    
    def get_ml_library_versions(self) -> Dict[str, Optional[str]]:
        """Получить версии ключевых ML библиотек.
        
        Returns:
            Словарь с версиями ML библиотек
        """
        versions = {}
        
        for lib_name in self.ml_libraries:
            try:
                # Пытаемся импортировать и получить версию
                if lib_name == 'torch':
                    import torch
                    versions[lib_name] = torch.__version__
                elif lib_name == 'stable-baselines3':
                    import stable_baselines3
                    versions[lib_name] = stable_baselines3.__version__
                elif lib_name == 'gymnasium':
                    import gymnasium
                    versions[lib_name] = gymnasium.__version__
                elif lib_name == 'numpy':
                    import numpy
                    versions[lib_name] = numpy.__version__
                elif lib_name == 'pandas':
                    import pandas
                    versions[lib_name] = pandas.__version__
                elif lib_name == 'matplotlib':
                    import matplotlib
                    versions[lib_name] = matplotlib.__version__
                elif lib_name == 'scikit-learn':
                    import sklearn
                    versions[lib_name] = sklearn.__version__
                else:
                    # Общий подход для других библиотек
                    module = __import__(lib_name.replace('-', '_'))
                    versions[lib_name] = getattr(module, '__version__', 'unknown')
                    
            except ImportError:
                versions[lib_name] = None
            except Exception as e:
                logger.debug(f"Ошибка получения версии {lib_name}: {e}")
                versions[lib_name] = 'error'
        
        return versions
    
    def create_dependency_snapshot(self, snapshot_name: Optional[str] = None) -> Dict[str, Any]:
        """Создать снимок зависимостей.
        
        Args:
            snapshot_name: Имя снимка (по умолчанию - timestamp)
            
        Returns:
            Словарь с информацией о снимке
        """
        if snapshot_name is None:
            snapshot_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Создание снимка зависимостей: {snapshot_name}")
        
        snapshot = {
            'metadata': {
                'name': snapshot_name,
                'timestamp': datetime.now().isoformat(),
                'project_root': str(self.project_root)
            },
            'system': self.get_system_info(),
            'package_managers': self.get_package_manager_info(),
            'packages': {
                'pip': self.get_pip_packages(),
                'conda': self.get_conda_packages()
            },
            'ml_libraries': self.get_ml_library_versions()
        }
        
        # Вычисляем хеш снимка для детектирования изменений
        snapshot_str = json.dumps(snapshot, sort_keys=True)
        snapshot['metadata']['hash'] = hashlib.sha256(snapshot_str.encode()).hexdigest()
        
        # Сохраняем снимок
        snapshot_file = self.snapshots_dir / f"snapshot_{snapshot_name}.json"
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Снимок сохранен: {snapshot_file}")
        return snapshot
    
    def load_snapshot(self, snapshot_name: str) -> Optional[Dict[str, Any]]:
        """Загрузить снимок зависимостей.
        
        Args:
            snapshot_name: Имя снимка
            
        Returns:
            Словарь с данными снимка или None если не найден
        """
        snapshot_file = self.snapshots_dir / f"snapshot_{snapshot_name}.json"
        
        if not snapshot_file.exists():
            logger.error(f"Снимок не найден: {snapshot_file}")
            return None
        
        try:
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Ошибка загрузки снимка {snapshot_name}: {e}")
            return None
    
    def compare_snapshots(self, snapshot1_name: str, snapshot2_name: str) -> Dict[str, Any]:
        """Сравнить два снимка зависимостей.
        
        Args:
            snapshot1_name: Имя первого снимка
            snapshot2_name: Имя второго снимка
            
        Returns:
            Словарь с результатами сравнения
        """
        snapshot1 = self.load_snapshot(snapshot1_name)
        snapshot2 = self.load_snapshot(snapshot2_name)
        
        if not snapshot1 or not snapshot2:
            raise ValueError("Один или оба снимка не найдены")
        
        comparison = {
            'metadata': {
                'snapshot1': snapshot1_name,
                'snapshot2': snapshot2_name,
                'comparison_time': datetime.now().isoformat()
            },
            'changes': {
                'packages_added': {},
                'packages_removed': {},
                'packages_updated': {},
                'ml_libraries_changed': {},
                'system_changes': {}
            }
        }
        
        # Сравниваем pip пакеты
        pip1 = snapshot1['packages']['pip']
        pip2 = snapshot2['packages']['pip']
        
        for pkg, version in pip2.items():
            if pkg not in pip1:
                comparison['changes']['packages_added'][pkg] = version
            elif pip1[pkg] != version:
                comparison['changes']['packages_updated'][pkg] = {
                    'old': pip1[pkg],
                    'new': version
                }
        
        for pkg in pip1:
            if pkg not in pip2:
                comparison['changes']['packages_removed'][pkg] = pip1[pkg]
        
        # Сравниваем ML библиотеки
        ml1 = snapshot1['ml_libraries']
        ml2 = snapshot2['ml_libraries']
        
        for lib, version in ml2.items():
            if lib in ml1 and ml1[lib] != version:
                comparison['changes']['ml_libraries_changed'][lib] = {
                    'old': ml1[lib],
                    'new': version
                }
        
        # Сравниваем системную информацию
        if snapshot1['system']['python']['version'] != snapshot2['system']['python']['version']:
            comparison['changes']['system_changes']['python_version'] = {
                'old': snapshot1['system']['python']['version'],
                'new': snapshot2['system']['python']['version']
            }
        
        return comparison
    
    def detect_dependency_conflicts(self) -> List[Dict[str, Any]]:
        """Детектировать потенциальные конфликты зависимостей.
        
        Returns:
            Список обнаруженных конфликтов
        """
        conflicts = []
        
        try:
            # Проверяем pip check
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'check'],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        conflicts.append({
                            'type': 'pip_dependency_conflict',
                            'description': line.strip()
                        })
        except subprocess.CalledProcessError:
            pass
        
        # Проверяем известные конфликты ML библиотек
        packages = self.get_pip_packages()
        
        # Конфликт между gym и gymnasium
        if 'gym' in packages and 'gymnasium' in packages:
            conflicts.append({
                'type': 'library_conflict',
                'description': 'Обнаружены одновременно gym и gymnasium. Рекомендуется использовать только gymnasium.',
                'packages': ['gym', 'gymnasium']
            })
        
        # Проверяем версии PyTorch и CUDA
        if 'torch' in packages:
            try:
                import torch
                if torch.cuda.is_available():
                    cuda_version = torch.version.cuda
                    if cuda_version and not self._is_cuda_compatible():
                        conflicts.append({
                            'type': 'cuda_version_mismatch',
                            'description': f'Версия CUDA PyTorch ({cuda_version}) может не соответствовать системной CUDA'
                        })
            except ImportError:
                pass
        
        return conflicts
    
    def _is_cuda_compatible(self) -> bool:
        """Проверить совместимость CUDA версий.
        
        Returns:
            True если версии совместимы
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True, text=True, check=True
            )
            return bool(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def generate_compatibility_report(self, snapshot_name: Optional[str] = None) -> Dict[str, Any]:
        """Генерировать отчет совместимости.
        
        Args:
            snapshot_name: Имя снимка для анализа (текущее состояние если None)
            
        Returns:
            Отчет совместимости
        """
        if snapshot_name:
            snapshot = self.load_snapshot(snapshot_name)
            if not snapshot:
                raise ValueError(f"Снимок {snapshot_name} не найден")
        else:
            snapshot = self.create_dependency_snapshot("temp_compatibility_check")
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'snapshot_name': snapshot_name or 'current'
            },
            'system_compatibility': {
                'python_version_ok': self._check_python_version(snapshot),
                'platform_supported': self._check_platform_support(snapshot),
                'memory_sufficient': self._check_memory_requirements(snapshot)
            },
            'package_compatibility': {
                'ml_libraries_compatible': self._check_ml_compatibility(snapshot),
                'dependency_conflicts': self.detect_dependency_conflicts()
            },
            'recommendations': []
        }
        
        # Генерируем рекомендации
        if not report['system_compatibility']['python_version_ok']:
            report['recommendations'].append(
                "Рекомендуется обновить Python до версии 3.8 или выше"
            )
        
        if report['package_compatibility']['dependency_conflicts']:
            report['recommendations'].append(
                "Обнаружены конфликты зависимостей. Рекомендуется их устранить"
            )
        
        if not report['system_compatibility']['memory_sufficient']:
            report['recommendations'].append(
                "Недостаточно оперативной памяти для комфортного обучения RL агентов"
            )
        
        return report
    
    def _check_python_version(self, snapshot: Dict[str, Any]) -> bool:
        """Проверить версию Python."""
        version_info = snapshot['system']['python']['version_info']
        return version_info['major'] >= 3 and version_info['minor'] >= 8
    
    def _check_platform_support(self, snapshot: Dict[str, Any]) -> bool:
        """Проверить поддержку платформы."""
        system = snapshot['system']['platform']['system']
        return system in ['Linux', 'Darwin', 'Windows']
    
    def _check_memory_requirements(self, snapshot: Dict[str, Any]) -> bool:
        """Проверить требования к памяти."""
        memory_gb = snapshot['system']['hardware']['memory_total'] / (1024**3)
        return memory_gb >= 4.0  # Минимум 4GB для RL
    
    def _check_ml_compatibility(self, snapshot: Dict[str, Any]) -> bool:
        """Проверить совместимость ML библиотек."""
        ml_libs = snapshot['ml_libraries']
        
        # Проверяем наличие ключевых библиотек
        required_libs = ['numpy', 'stable-baselines3', 'gymnasium']
        for lib in required_libs:
            if not ml_libs.get(lib):
                return False
        
        return True
    
    def export_requirements(self, format_type: str = 'pip', 
                          output_file: Optional[Union[str, Path]] = None) -> str:
        """Экспортировать зависимости в различные форматы.
        
        Args:
            format_type: Формат экспорта ('pip', 'conda', 'poetry')
            output_file: Файл для сохранения (опционально)
            
        Returns:
            Строка с зависимостями
        """
        if format_type == 'pip':
            return self._export_pip_requirements(output_file)
        elif format_type == 'conda':
            return self._export_conda_environment(output_file)
        elif format_type == 'poetry':
            return self._export_poetry_requirements(output_file)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format_type}")
    
    def _export_pip_requirements(self, output_file: Optional[Union[str, Path]]) -> str:
        """Экспортировать pip requirements."""
        packages = self.get_pip_packages()
        
        lines = []
        for package, version in sorted(packages.items()):
            lines.append(f"{package}=={version}")
        
        content = '\n'.join(lines)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Requirements сохранены в: {output_path}")
        
        return content
    
    def _export_conda_environment(self, output_file: Optional[Union[str, Path]]) -> str:
        """Экспортировать conda environment."""
        try:
            result = subprocess.run(
                ['conda', 'env', 'export'],
                capture_output=True, text=True, check=True
            )
            content = result.stdout
            
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Conda environment сохранен в: {output_path}")
            
            return content
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Ошибка экспорта conda environment: {e}")
            return ""
    
    def _export_poetry_requirements(self, output_file: Optional[Union[str, Path]]) -> str:
        """Экспортировать poetry requirements."""
        # Базовая реализация - можно расширить
        packages = self.get_pip_packages()
        
        poetry_content = {
            'tool': {
                'poetry': {
                    'name': 'rl-training',
                    'version': '0.1.0',
                    'description': 'RL Training Dependencies',
                    'dependencies': {
                        'python': '^3.8'
                    }
                }
            }
        }
        
        for package, version in packages.items():
            poetry_content['tool']['poetry']['dependencies'][package] = f"^{version}"
        
        content = yaml.dump(poetry_content, default_flow_style=False)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Poetry pyproject.toml сохранен в: {output_path}")
        
        return content
    
    def validate_reproducibility(self, reference_snapshot: str) -> Dict[str, Any]:
        """Валидировать воспроизводимость среды.
        
        Args:
            reference_snapshot: Имя эталонного снимка
            
        Returns:
            Результат валидации
        """
        current_snapshot = self.create_dependency_snapshot("validation_temp")
        reference = self.load_snapshot(reference_snapshot)
        
        if not reference:
            raise ValueError(f"Эталонный снимок {reference_snapshot} не найден")
        
        validation = {
            'timestamp': datetime.now().isoformat(),
            'reference_snapshot': reference_snapshot,
            'reproducible': True,
            'issues': [],
            'warnings': []
        }
        
        # Проверяем версию Python
        ref_python = reference['system']['python']['version_info']
        cur_python = current_snapshot['system']['python']['version_info']
        
        if ref_python != cur_python:
            validation['reproducible'] = False
            validation['issues'].append({
                'type': 'python_version_mismatch',
                'reference': f"{ref_python['major']}.{ref_python['minor']}.{ref_python['micro']}",
                'current': f"{cur_python['major']}.{cur_python['minor']}.{cur_python['micro']}"
            })
        
        # Проверяем ключевые ML библиотеки
        ref_ml = reference['ml_libraries']
        cur_ml = current_snapshot['ml_libraries']
        
        for lib in self.ml_libraries:
            if ref_ml.get(lib) != cur_ml.get(lib):
                if ref_ml.get(lib) and not cur_ml.get(lib):
                    validation['reproducible'] = False
                    validation['issues'].append({
                        'type': 'missing_library',
                        'library': lib,
                        'reference_version': ref_ml[lib]
                    })
                elif ref_ml.get(lib) != cur_ml.get(lib):
                    validation['warnings'].append({
                        'type': 'version_mismatch',
                        'library': lib,
                        'reference_version': ref_ml.get(lib),
                        'current_version': cur_ml.get(lib)
                    })
        
        # Проверяем платформу
        ref_platform = reference['system']['platform']['system']
        cur_platform = current_snapshot['system']['platform']['system']
        
        if ref_platform != cur_platform:
            validation['warnings'].append({
                'type': 'platform_mismatch',
                'reference': ref_platform,
                'current': cur_platform
            })
        
        return validation
    
    def get_snapshots_list(self) -> List[Dict[str, Any]]:
        """Получить список всех снимков.
        
        Returns:
            Список снимков с метаданными
        """
        snapshots = []
        
        for snapshot_file in self.snapshots_dir.glob("snapshot_*.json"):
            try:
                with open(snapshot_file, 'r', encoding='utf-8') as f:
                    snapshot = json.load(f)
                    snapshots.append({
                        'name': snapshot['metadata']['name'],
                        'timestamp': snapshot['metadata']['timestamp'],
                        'file': str(snapshot_file),
                        'hash': snapshot['metadata'].get('hash', 'unknown')
                    })
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Ошибка чтения снимка {snapshot_file}: {e}")
        
        return sorted(snapshots, key=lambda x: x['timestamp'], reverse=True)
    
    def cleanup_old_snapshots(self, keep_count: int = 10) -> int:
        """Очистить старые снимки.
        
        Args:
            keep_count: Количество снимков для сохранения
            
        Returns:
            Количество удаленных снимков
        """
        snapshots = self.get_snapshots_list()
        
        if len(snapshots) <= keep_count:
            return 0
        
        to_delete = snapshots[keep_count:]
        deleted_count = 0
        
        for snapshot in to_delete:
            try:
                Path(snapshot['file']).unlink()
                deleted_count += 1
                logger.debug(f"Удален снимок: {snapshot['name']}")
            except OSError as e:
                logger.warning(f"Ошибка удаления снимка {snapshot['name']}: {e}")
        
        logger.info(f"Удалено {deleted_count} старых снимков")
        return deleted_count


def create_experiment_snapshot(experiment_id: str, 
                             project_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Создать снимок зависимостей для эксперимента.
    
    Args:
        experiment_id: Идентификатор эксперимента
        project_root: Корневая директория проекта
        
    Returns:
        Словарь с информацией о снимке
    """
    tracker = DependencyTracker(project_root)
    snapshot_name = f"experiment_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    snapshot = tracker.create_dependency_snapshot(snapshot_name)
    snapshot['metadata']['experiment_id'] = experiment_id
    
    # Пересохраняем с experiment_id
    snapshot_file = tracker.snapshots_dir / f"snapshot_{snapshot_name}.json"
    with open(snapshot_file, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Создан снимок для эксперимента {experiment_id}: {snapshot_name}")
    return snapshot


def validate_environment_for_experiment(reference_snapshot: str,
                                       project_root: Optional[Union[str, Path]] = None) -> bool:
    """Валидировать среду для воспроизведения эксперимента.
    
    Args:
        reference_snapshot: Имя эталонного снимка
        project_root: Корневая директория проекта
        
    Returns:
        True если среда воспроизводима
    """
    tracker = DependencyTracker(project_root)
    validation = tracker.validate_reproducibility(reference_snapshot)
    
    if not validation['reproducible']:
        logger.error("Среда не воспроизводима!")
        for issue in validation['issues']:
            logger.error(f"Проблема: {issue}")
        return False
    
    if validation['warnings']:
        logger.warning("Обнаружены предупреждения:")
        for warning in validation['warnings']:
            logger.warning(f"Предупреждение: {warning}")
    
    logger.info("Среда валидна для воспроизведения эксперимента")
    return True