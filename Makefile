# Makefile для проекта обучения RL агентов
# Упрощает запуск тестов, линтинга и других задач разработки

.PHONY: help test test-unit test-integration test-slow test-cli lint format clean install dev-install

# Переменные
PYTHON := python
PIP := pip
PYTEST := pytest
RUFF := ruff
MYPY := mypy

# Директории
SRC_DIR := src
TEST_DIR := tests
INTEGRATION_TEST_DIR := tests/integration

# Цвета для вывода
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Показать справку по командам
	@echo "$(GREEN)Доступные команды:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# Установка зависимостей
install: ## Установить основные зависимости
	$(PIP) install -r requirements.txt

dev-install: ## Установить зависимости для разработки
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PIP) install pytest pytest-cov pytest-mock pytest-timeout ruff mypy

# Тестирование
test: ## Запустить все тесты
	@echo "$(GREEN)Запуск всех тестов...$(NC)"
	$(PYTEST) $(TEST_DIR) -v

test-unit: ## Запустить только юнит-тесты
	@echo "$(GREEN)Запуск юнит-тестов...$(NC)"
	$(PYTEST) $(TEST_DIR) -m "not integration and not slow" -v

test-integration: ## Запустить интеграционные тесты
	@echo "$(GREEN)Запуск интеграционных тестов...$(NC)"
	$(PYTEST) $(INTEGRATION_TEST_DIR) -m integration -v --timeout=300

test-integration-fast: ## Запустить быстрые интеграционные тесты
	@echo "$(GREEN)Запуск быстрых интеграционных тестов...$(NC)"
	$(PYTEST) $(INTEGRATION_TEST_DIR) -m "integration and not slow" -v --timeout=120

test-slow: ## Запустить медленные тесты (с реальным обучением)
	@echo "$(YELLOW)Запуск медленных тестов (может занять много времени)...$(NC)"
	$(PYTEST) $(TEST_DIR) -m slow -v --timeout=600

test-cli: ## Запустить тесты CLI интерфейса
	@echo "$(GREEN)Запуск тестов CLI...$(NC)"
	$(PYTEST) $(INTEGRATION_TEST_DIR)/test_cli_interface.py -v

test-controlled-experiments: ## Запустить тесты контролируемых экспериментов
	@echo "$(GREEN)Запуск тестов контролируемых экспериментов...$(NC)"
	$(PYTEST) $(INTEGRATION_TEST_DIR)/test_controlled_experiments.py -v

test-with-coverage: ## Запустить тесты с покрытием кода
	@echo "$(GREEN)Запуск тестов с анализом покрытия...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

test-specific: ## Запустить конкретный тест (использование: make test-specific TEST=test_name)
	@echo "$(GREEN)Запуск теста: $(TEST)$(NC)"
	$(PYTEST) -k "$(TEST)" -v

# Линтинг и форматирование
lint: ## Проверить код с помощью ruff
	@echo "$(GREEN)Проверка кода с помощью ruff...$(NC)"
	$(RUFF) check $(SRC_DIR) $(TEST_DIR)

lint-fix: ## Исправить проблемы кода автоматически
	@echo "$(GREEN)Автоматическое исправление проблем кода...$(NC)"
	$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR)

format: ## Форматировать код
	@echo "$(GREEN)Форматирование кода...$(NC)"
	$(RUFF) format $(SRC_DIR) $(TEST_DIR)

type-check: ## Проверить типы с помощью mypy
	@echo "$(GREEN)Проверка типов...$(NC)"
	$(MYPY) $(SRC_DIR) --strict

# Комплексные проверки
check-all: lint type-check test-unit ## Выполнить все проверки кода
	@echo "$(GREEN)Все проверки завершены!$(NC)"

ci-test: ## Тесты для CI/CD (быстрые)
	@echo "$(GREEN)Запуск тестов для CI/CD...$(NC)"
	$(PYTEST) $(TEST_DIR) -m "not slow" --timeout=120 --cov=$(SRC_DIR) --cov-report=xml

# Очистка
clean: ## Очистить временные файлы
	@echo "$(GREEN)Очистка временных файлов...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf results/tests
	rm -rf results/tmp

clean-all: clean ## Полная очистка (включая результаты экспериментов)
	@echo "$(YELLOW)Полная очистка...$(NC)"
	rm -rf results/experiments
	rm -rf results/models
	rm -rf results/logs

# Запуск экспериментов
run-test-experiment: ## Запустить тестовый эксперимент PPO vs A2C
	@echo "$(GREEN)Запуск тестового эксперимента...$(NC)"
	$(PYTHON) -m src.experiments.runner \
		--config configs/test_ppo_vs_a2c.yaml \
		--mode validation \
		--verbose

run-full-experiment: ## Запустить полный эксперимент PPO vs A2C
	@echo "$(YELLOW)Запуск полного эксперимента (может занять много времени)...$(NC)"
	$(PYTHON) -m src.experiments.runner \
		--config configs/ppo_vs_a2c_experiment.yaml \
		--mode sequential \
		--verbose

# Разработка
dev-setup: dev-install ## Настроить среду разработки
	@echo "$(GREEN)Настройка среды разработки...$(NC)"
	pre-commit install || echo "$(YELLOW)pre-commit не установлен, пропускаем...$(NC)"

dev-test: ## Быстрые тесты для разработки
	@echo "$(GREEN)Быстрые тесты для разработки...$(NC)"
	$(PYTEST) $(TEST_DIR) -x -v --timeout=60

# Документация
docs: ## Генерировать документацию
	@echo "$(GREEN)Генерация документации...$(NC)"
	@echo "$(YELLOW)Документация пока не настроена$(NC)"

# Профилирование
profile-tests: ## Профилировать тесты
	@echo "$(GREEN)Профилирование тестов...$(NC)"
	$(PYTEST) $(TEST_DIR) --profile --profile-svg

# Безопасность
security-check: ## Проверить безопасность зависимостей
	@echo "$(GREEN)Проверка безопасности...$(NC)"
	$(PIP) audit || echo "$(YELLOW)pip-audit не установлен$(NC)"

# Информация о проекте
info: ## Показать информацию о проекте
	@echo "$(GREEN)Информация о проекте:$(NC)"
	@echo "Python версия: $$($(PYTHON) --version)"
	@echo "Pip версия: $$($(PIP) --version)"
	@echo "Pytest версия: $$($(PYTEST) --version)"
	@echo "Ruff версия: $$($(RUFF) --version)"
	@echo "Директория проекта: $$(pwd)"
	@echo "Количество Python файлов в src: $$(find $(SRC_DIR) -name '*.py' | wc -l)"
	@echo "Количество тестовых файлов: $$(find $(TEST_DIR) -name 'test_*.py' | wc -l)"

# Примеры использования
examples: ## Показать примеры использования
	@echo "$(GREEN)Примеры использования:$(NC)"
	@echo "  $(YELLOW)make test-unit$(NC)                    # Быстрые юнит-тесты"
	@echo "  $(YELLOW)make test-integration-fast$(NC)        # Быстрые интеграционные тесты"
	@echo "  $(YELLOW)make test-controlled-experiments$(NC)  # Тесты экспериментов"
	@echo "  $(YELLOW)make lint-fix$(NC)                     # Исправить код автоматически"
	@echo "  $(YELLOW)make run-test-experiment$(NC)          # Запустить тестовый эксперимент"
	@echo "  $(YELLOW)make check-all$(NC)                    # Все проверки кода"
	@echo "  $(YELLOW)make clean$(NC)                        # Очистить временные файлы"

# Алиасы для удобства
t: test-unit ## Алиас для test-unit
ti: test-integration-fast ## Алиас для test-integration-fast
l: lint ## Алиас для lint
f: format ## Алиас для format
c: clean ## Алиас для clean

# По умолчанию показываем справку
.DEFAULT_GOAL := help