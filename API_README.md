# RL Agent Training API

FastAPI приложение для управления экспериментами обучения агентов подкрепляющего обучения (RL).

## 🚀 Быстрый старт

### Установка зависимостей

```bash
# Установка FastAPI и зависимостей
pip install fastapi uvicorn pydantic python-multipart

# Или если используете requirements.txt
pip install -r requirements.txt
```

### Запуск сервера

```bash
# Простой запуск
python -m src.api.app

# С параметрами
python -m src.api.app --host 0.0.0.0 --port 8000 --debug

# Или через uvicorn напрямую
uvicorn src.api.app:create_app --host 0.0.0.0 --port 8000 --reload
```

### Тестирование

```bash
# Запуск тестов API
python test_api.py
```

## 📚 Документация API

После запуска сервера документация доступна по адресам:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🔧 Конфигурация

### Переменные окружения

Скопируйте `.env.example` в `.env` и настройте под свои нужды:

```bash
cp .env.example .env
```

Основные переменные:

```bash
# Сервер
RL_API_HOST=0.0.0.0
RL_API_PORT=8000
RL_API_DEBUG=false

# CORS
RL_API_CORS_ORIGINS="http://localhost:3000,http://localhost:8080"

# Логирование
RL_API_LOG_LEVEL="INFO"
RL_API_LOG_DIR="logs/api"

# RL система
RL_API_MAX_CONCURRENT_EXPERIMENTS=3
RL_API_DEFAULT_TIMEOUT_MINUTES=60
```

### Программная конфигурация

```python
from src.api.config import create_api_config, setup_api_config

# Создание кастомной конфигурации
config = create_api_config(
    base_dir="/path/to/project",
    max_concurrent_experiments=5
)

# Установка глобальной конфигурации
setup_api_config(config)
```

## 📋 API Эндпоинты

### Эксперименты

#### Создание эксперимента
```http
POST /experiments
Content-Type: application/json

{
  "name": "test_ppo_lunarlander",
  "algorithm": "PPO",
  "environment": "LunarLander-v2",
  "hyperparameters": {
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64
  },
  "seed": 42,
  "description": "Тестирование PPO на LunarLander",
  "hypothesis": "PPO должен показать хорошие результаты на LunarLander"
}
```

#### Список экспериментов
```http
GET /experiments?status=running&algorithm=PPO
```

#### Информация об эксперименте
```http
GET /experiments/{experiment_id}
```

#### Обновление эксперимента
```http
PUT /experiments/{experiment_id}
Content-Type: application/json

{
  "status": "paused",
  "hyperparameters": {
    "learning_rate": 0.0001
  }
}
```

### Обучение

#### Запуск обучения
```http
POST /experiments/{experiment_id}/train
```

#### Получение метрик
```http
GET /experiments/{experiment_id}/metrics?from_timestep=1000&to_timestep=5000
```

#### Получение результатов
```http
GET /experiments/{experiment_id}/results
```

### Метаданные

#### Список сред
```http
GET /environments
```

#### Список алгоритмов
```http
GET /algorithms
```

### Служебные

#### Healthcheck
```http
GET /health
```

#### Конфигурация
```http
GET /config
```

## 🏗️ Архитектура

### Структура файлов

```
src/api/
├── __init__.py          # Экспорты модуля
├── app.py              # Основное FastAPI приложение
├── config.py           # Конфигурация API
└── dependencies.py     # Зависимости и сервисы
```

### Компоненты

#### APIConfig
Центральная конфигурация API с настройками сервера, CORS, логирования и интеграции с RL системой.

#### Сервисы
- **ExperimentService**: Управление экспериментами
- **EnvironmentService**: Информация о средах RL
- **AlgorithmService**: Информация об алгоритмах RL

#### Middleware
- **CORS**: Настройка кросс-доменных запросов
- **TrustedHost**: Проверка доверенных хостов
- **Request Logging**: Логирование HTTP запросов

#### Обработка ошибок
- Кастомные обработчики для HTTP и общих исключений
- Структурированные ответы об ошибках
- Логирование ошибок с контекстом

## 🔒 Безопасность

### Аутентификация (опционально)

API поддерживает Bearer токены для аутентификации:

```http
Authorization: Bearer your-token-here
```

Для тестирования используйте токен `test-token`.

### CORS

Настройте разрешенные домены в переменных окружения:

```bash
RL_API_CORS_ORIGINS="http://localhost:3000,https://yourdomain.com"
```

## 📊 Мониторинг

### Healthcheck

Эндпоинт `/health` возвращает:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-14T12:00:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "active_experiments": 2
}
```

### Логирование

API логирует:
- HTTP запросы и ответы
- Операции с экспериментами
- Ошибки и исключения
- Системные события

Логи сохраняются в директории `logs/api/` в JSON формате.

### Метрики

Время обработки запросов добавляется в заголовок `X-Process-Time`.

## 🔄 Graceful Shutdown

API корректно завершает работу:
1. Перехватывает сигналы SIGINT/SIGTERM
2. Ожидает завершения активных экспериментов (до 30 сек)
3. Закрывает соединения и освобождает ресурсы

## 🧪 Примеры использования

### Python клиент

```python
import requests

# Создание эксперимента
response = requests.post("http://localhost:8000/experiments", json={
    "name": "my_experiment",
    "algorithm": "PPO",
    "environment": "LunarLander-v2",
    "hyperparameters": {"learning_rate": 0.0003},
    "seed": 42
})

experiment = response.json()
experiment_id = experiment["id"]

# Запуск обучения
requests.post(f"http://localhost:8000/experiments/{experiment_id}/train")

# Получение метрик
metrics = requests.get(f"http://localhost:8000/experiments/{experiment_id}/metrics")
print(metrics.json())
```

### cURL

```bash
# Создание эксперимента
curl -X POST "http://localhost:8000/experiments" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "curl_test",
    "algorithm": "PPO",
    "environment": "LunarLander-v2",
    "seed": 42
  }'

# Список экспериментов
curl "http://localhost:8000/experiments"

# Healthcheck
curl "http://localhost:8000/health"
```

## 🐛 Отладка

### Режим отладки

```bash
python -m src.api.app --debug --reload
```

В режиме отладки:
- Включено подробное логирование
- Автоперезагрузка при изменениях кода
- Детальная информация об ошибках

### Логи

Проверьте логи в директории `logs/api/`:
- `training_*.log` - основные логи
- `errors_*.log` - только ошибки

### Тестирование

```bash
# Простой тест конфигурации
python test_api.py

# Проверка эндпоинтов
curl http://localhost:8000/health
curl http://localhost:8000/environments
curl http://localhost:8000/algorithms
```

## 🚀 Развертывание

### Продакшен

```bash
# С несколькими workers
python -m src.api.app --workers 4 --host 0.0.0.0 --port 8000

# Или через gunicorn
pip install gunicorn
gunicorn src.api.app:create_app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "-m", "src.api.app", "--host", "0.0.0.0", "--port", "8000"]
```

### Переменные окружения для продакшена

```bash
RL_API_DEBUG=false
RL_API_SECRET_KEY="your-secure-secret-key"
RL_API_CORS_ORIGINS="https://yourdomain.com"
RL_API_LOG_LEVEL="WARNING"
RL_API_WORKERS=4
```

## 📝 Лицензия

Этот проект является частью системы обучения RL агентов МИФИ.

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## 📞 Поддержка

Для вопросов и предложений создайте Issue в репозитории проекта.