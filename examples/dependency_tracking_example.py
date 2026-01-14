#!/usr/bin/env python3
"""Пример использования системы отслеживания зависимостей.

Этот скрипт демонстрирует основные возможности DependencyTracker
для обеспечения воспроизводимости RL экспериментов.
"""

from pathlib import Path

from src.utils.dependency_tracker import (
    DependencyTracker,
    create_experiment_snapshot,
    validate_environment_for_experiment,
)
from src.utils.logging import setup_logging

# Настройка логирования
logger = setup_logging(log_level="INFO", console_output=True)


def main():
    """Демонстрация функций отслеживания зависимостей."""
    print("🔍 Демонстрация системы отслеживания зависимостей")
    print("=" * 60)
    
    # Инициализация трекера
    tracker = DependencyTracker()
    
    # 1. Получение информации о системе
    print("\n1. 📊 Информация о системе:")
    system_info = tracker.get_system_info()
    print(f"   Python: {system_info['python']['version']}")
    print(f"   Платформа: {system_info['platform']['system']} {system_info['platform']['release']}")
    print(f"   Память: {system_info['hardware']['memory_total'] / (1024**3):.1f} GB")
    print(f"   CPU: {system_info['hardware']['cpu_count']} ядер")
    
    # 2. Информация о менеджерах пакетов
    print("\n2. 📦 Менеджеры пакетов:")
    managers = tracker.get_package_manager_info()
    for manager, info in managers.items():
        status = "✅ доступен" if info['available'] else "❌ недоступен"
        print(f"   {manager}: {status}")
        if info['available'] and 'version' in info:
            print(f"      Версия: {info['version']}")
    
    # 3. ML библиотеки
    print("\n3. 🤖 ML библиотеки:")
    ml_versions = tracker.get_ml_library_versions()
    for lib, version in ml_versions.items():
        if version:
            status = f"✅ {version}"
        elif version is None:
            status = "❌ не установлена"
        else:
            status = "⚠️ ошибка определения версии"
        print(f"   {lib}: {status}")
    
    # 4. Создание снимка зависимостей
    print("\n4. 📸 Создание снимка зависимостей:")
    snapshot = tracker.create_dependency_snapshot("demo_snapshot")
    print(f"   Снимок создан: {snapshot['metadata']['name']}")
    print(f"   Временная метка: {snapshot['metadata']['timestamp']}")
    print(f"   Хеш: {snapshot['metadata']['hash'][:16]}...")
    
    # 5. Детектирование конфликтов
    print("\n5. ⚠️ Проверка конфликтов зависимостей:")
    conflicts = tracker.detect_dependency_conflicts()
    if conflicts:
        print(f"   Найдено {len(conflicts)} конфликтов:")
        for conflict in conflicts:
            print(f"   - {conflict['type']}: {conflict['description']}")
    else:
        print("   ✅ Конфликтов не обнаружено")
    
    # 6. Отчет совместимости
    print("\n6. 📋 Отчет совместимости:")
    report = tracker.generate_compatibility_report()
    compatibility = report['system_compatibility']
    print(f"   Python версия: {'✅' if compatibility['python_version_ok'] else '❌'}")
    print(f"   Платформа: {'✅' if compatibility['platform_supported'] else '❌'}")
    print(f"   Память: {'✅' if compatibility['memory_sufficient'] else '❌'}")
    
    package_compat = report['package_compatibility']
    print(f"   ML библиотеки: {'✅' if package_compat['ml_libraries_compatible'] else '❌'}")
    
    if report['recommendations']:
        print("   📝 Рекомендации:")
        for rec in report['recommendations']:
            print(f"   - {rec}")
    
    # 7. Экспорт зависимостей
    print("\n7. 💾 Экспорт зависимостей:")
    
    # Экспорт pip requirements
    requirements_file = Path("requirements_demo.txt")
    pip_content = tracker.export_requirements('pip', requirements_file)
    print(f"   pip requirements: {requirements_file} ({len(pip_content.split())} пакетов)")
    
    # Экспорт conda environment (если доступен)
    conda_file = None
    if managers['conda']['available']:
        conda_file = Path("environment_demo.yml")
        tracker.export_requirements('conda', conda_file)
        print(f"   conda environment: {conda_file}")
    
    # 8. Список снимков
    print("\n8. 📂 Список снимков:")
    snapshots = tracker.get_snapshots_list()
    print(f"   Всего снимков: {len(snapshots)}")
    for snap in snapshots[:3]:  # Показываем первые 3
        print(f"   - {snap['name']} ({snap['timestamp']})")
    
    # 9. Демонстрация создания снимка для эксперимента
    print("\n9. 🧪 Снимок для эксперимента:")
    exp_snapshot = create_experiment_snapshot("demo_experiment_001")
    print(f"   Создан снимок для эксперимента: {exp_snapshot['metadata']['experiment_id']}")
    
    # 10. Валидация воспроизводимости
    print("\n10. ✅ Валидация воспроизводимости:")
    try:
        is_valid = validate_environment_for_experiment("demo_snapshot")
        print(f"    Среда воспроизводима: {'✅ Да' if is_valid else '❌ Нет'}")
    except ValueError as e:
        print(f"    ⚠️ Ошибка валидации: {e}")
    
    # Очистка демонстрационных файлов
    print("\n🧹 Очистка демонстрационных файлов...")
    if requirements_file.exists():
        requirements_file.unlink()
        print(f"   Удален: {requirements_file}")
    
    if conda_file and conda_file.exists():
        conda_file.unlink()
        print(f"   Удален: {conda_file}")
    
    print("\n✨ Демонстрация завершена!")


def demonstrate_snapshot_comparison():
    """Демонстрация сравнения снимков."""
    print("\n" + "=" * 60)
    print("🔄 Демонстрация сравнения снимков")
    print("=" * 60)
    
    tracker = DependencyTracker()
    
    # Создаем два снимка с небольшим интервалом
    print("Создание первого снимка...")
    tracker.create_dependency_snapshot("comparison_test_1")
    
    print("Создание второго снимка...")
    tracker.create_dependency_snapshot("comparison_test_2")
    
    # Сравниваем снимки
    print("Сравнение снимков...")
    try:
        comparison = tracker.compare_snapshots("comparison_test_1", "comparison_test_2")
        
        changes = comparison['changes']
        total_changes = (
            len(changes['packages_added']) +
            len(changes['packages_removed']) +
            len(changes['packages_updated']) +
            len(changes['ml_libraries_changed'])
        )
        
        print(f"Результат сравнения: {total_changes} изменений")
        
        if changes['packages_added']:
            print(f"  Добавлено пакетов: {len(changes['packages_added'])}")
        
        if changes['packages_removed']:
            print(f"  Удалено пакетов: {len(changes['packages_removed'])}")
        
        if changes['packages_updated']:
            print(f"  Обновлено пакетов: {len(changes['packages_updated'])}")
        
        if changes['ml_libraries_changed']:
            print(f"  Изменено ML библиотек: {len(changes['ml_libraries_changed'])}")
        
        if total_changes == 0:
            print("  ✅ Снимки идентичны")
    
    except Exception as e:
        print(f"  ❌ Ошибка сравнения: {e}")


def demonstrate_export_formats():
    """Демонстрация различных форматов экспорта."""
    print("\n" + "=" * 60)
    print("📤 Демонстрация форматов экспорта")
    print("=" * 60)
    
    tracker = DependencyTracker()
    
    # Получаем несколько пакетов для демонстрации
    packages = tracker.get_pip_packages()
    sample_packages = dict(list(packages.items())[:5])  # Первые 5 пакетов
    
    print(f"Демонстрация на примере {len(sample_packages)} пакетов:")
    for pkg, version in sample_packages.items():
        print(f"  - {pkg}=={version}")
    
    # Демонстрация различных форматов
    formats = ['pip', 'conda', 'poetry']
    
    for fmt in formats:
        print(f"\n📋 Формат {fmt}:")
        try:
            content = tracker.export_requirements(fmt)
            lines = content.split('\n')[:3]  # Первые 3 строки
            for line in lines:
                if line.strip():
                    print(f"  {line}")
            if len(content.split('\n')) > 3:
                print(f"  ... (еще {len(content.split('\n')) - 3} строк)")
        except Exception as e:
            print(f"  ❌ Ошибка экспорта: {e}")


if __name__ == "__main__":
    try:
        main()
        demonstrate_snapshot_comparison()
        demonstrate_export_formats()
    except KeyboardInterrupt:
        print("\n\n⏹️ Демонстрация прервана пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()