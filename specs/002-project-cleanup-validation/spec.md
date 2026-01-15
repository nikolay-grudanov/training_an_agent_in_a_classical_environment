# Feature Specification: Project Cleanup & PPO vs A2C Experiments

**Feature Branch**: `002-project-cleanup-validation`
**Created**: 2026-01-15
**Status**: Draft
**Input**: Очистить проект и провести контролируемые эксперименты PPO vs A2C на LunarLander-v3

## User Scenarios & Testing

### User Story 1 - Рабочий MVP: PPO обучение (Priority: P1)

Как исследователь, я хочу обучить PPO агента на LunarLander-v3, чтобы получить baseline результаты.

**Independent Test**: `python train_ppo.py --steps 50000` успешно завершается

**Acceptance Scenarios**:
1. **Given** скрипт `train_ppo.py` создан, **When** запускаю обучение с `--steps 50000`, **Then** обучение завершается без ошибок
2. **Given** обучение завершено, **When** проверяю выходные файлы, **Then** вижу: модель (`.zip`), метрики (`metrics.json`), конфигурацию (`config.json`)

---

### User Story 2 - Рабочий MVP: A2C обучение (Priority: P1)

Как исследователь, я хочу обучить A2C агента на LunarLander-v3, чтобы сравнить с PPO.

**Independent Test**: `python train_a2c.py --steps 50000` успешно завершается

**Acceptance Scenarios**:
1. **Given** скрипт `train_a2c.py` создан, **When** запускаю обучение с `--steps 50000`, **Then** обучение завершается без ошибок
2. **Given** обучение завершено, **When** проверяю выходные файлы, **Then** вижу: модель (`.zip`), метрики (`metrics.json`), конфигурацию (`config.json`)

---

### User Story 3 - Сравнение алгоритмов (Priority: P1)

Как исследователь, я хочу сравнить PPO и A2C на одной среде, чтобы выбрать лучший алгоритм.

**Independent Test**: Сравнительный отчёт показывает метрики обоих алгоритмов

**Acceptance Scenarios**:
1. **Given** оба эксперимента завершены с `--seed 42`, **When** сравниваю результаты, **Then** вижу таблицу с: final reward, training time, steps to threshold
2. **Given** результаты получены, **When** анализирую данные, **Then** делаю вывод о преимуществах каждого алгоритма

---

### User Story 4 - Очистка проекта (Priority: P2)

Как maintainer, я хочу оставить в проекте только необходимые файлы.

**Acceptance Scenarios**:
1. **Given** проект содержит лишние файлы в корне, **When** очищаю, **Then** остаются: `requirements.txt`, `README.md`, `.gitignore`, `src/`, `tests/`, `results/`
2. **Given** требования к зависимостям, **When** создаю `requirements.txt`, **Then** он содержит: stable-baselines3, gymnasium, numpy, torch, pyyaml

---

## Edge Cases

- Что если LunarLander-v3 недоступен? → Использовать `LunarLander-v2` с предупреждением
- Что если обучение прерывается? → Сохранять чекпоинты каждые 10K шагов
- Что если метрики не записываются? → Логировать в консоль как fallback

## Requirements

### Functional Requirements

- **FR-001**: Проект ДОЛЖЕН содержать рабочий скрипт `train_ppo.py`
- **FR-002**: Проект ДОЛЖЕН содержать рабочий скрипт `train_a2c.py`
- **FR-003**: Оба алгоритма ДОЛЖНЫ использовать `seed=42` для воспроизводимости
- **FR-004**: Среда ДОЛЖНА быть `LunarLander-v3` для обоих экспериментов
- **FR-005**: Каждый эксперимент ДОЛЖЕН сохранять: модель, метрики, конфигурацию
- **FR-006**: Эксперименты ДОЛЖНЫ запускаться с `--total_timesteps=50000`
- **FR-007**: Результаты ДОЛЖНЫ быть сохранены в `results/experiments/{ppo|a2c}_seed42/`
- **FR-008**: Проект ДОЛЖЕН содержать `requirements.txt` с зависимостями
- **FR-009**: Проект ДОЛЖЕН содержать `.gitignore` с правильными паттернами
- **FR-010**: Проект ДОЛЖЕН содержать `README.md` с инструкциями

### Out of Scope

- Реализация SAC, TD3, DQN (только PPO и A2C)
- API сервер
- Веб-визуализация
- Автоматическое сравнение (только ручное)

## Success Criteria

- **SC-001**: `train_ppo.py` успешно обучает агента на 50K+ шагов
- **SC-002**: `train_a2c.py` успешно обучает агента на 50K+ шагов
- **SC-003**: Обе модели сохранены в `results/experiments/`
- **SC-004**: Метрики записаны в `results/experiments/*/metrics.json`
- **SC-005**: `requirements.txt` содержит все зависимости
- **SC-006**: `.gitignore` исключает `__pycache__/`, `*.pyc`, `results/`, `.venv/`
- **SC-007**: `README.md` содержит инструкции по запуску
- **SC-008**: Сравнительный отчёт создан в `RESULTS.md`

## Assumptions

- **A-001**: Используем Stable-Baselines3 для обоих алгоритмов
- **A-002**: Gymnasium с `LunarLander-v3` доступен
- **A-003**: CPU достаточно для обучения (GPU не требуется)
- **A-004**: Seed=42 даёт воспроизводимые результаты
- **A-005**: Каждый эксперимент занимает не более 30 минут
