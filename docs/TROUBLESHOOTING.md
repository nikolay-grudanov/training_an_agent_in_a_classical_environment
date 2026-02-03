# TROUBLESHOOTING.md - Решение проблем 🛠️

Этот документ поможет вам решить распространённые проблемы при работе с проектом.

---

## 🐛 Общие проблемы

### 1. Проблема: `conda: command not found`

**Симптомы:**
```bash
conda activate rocm
# conda: command not found
```

**Решения:**

**Вариант A (bash):**
```bash
# Инициализируйте conda для bash
source ~/anaconda3/etc/profile.d/conda.sh
# или
source ~/miniconda3/etc/profile.d/conda.sh

# Затем активируйте
conda activate rocm
```

**Вариант B (zsh):**
```bash
# Инициализируйте conda для zsh
source ~/anaconda3/etc/profile.d/conda.d/conda.sh
conda activate rocm
```

**Вариант C (используйте полный путь):**
```bash
~/anaconda3/envs/rocm/bin/python run_experiments.py
# или
/opt/conda/envs/rocm/bin/python run_experiments.py
```

---

### 2. Проблема: `ModuleNotFoundError: No module named 'X'`

**Симптомы:**
```bash
python run_experiments.py
# ModuleNotFoundError: No module named 'stable_baselines3'
# ModuleNotFoundError: No module named 'gymnasium'
# etc.
```

**Решения:**

**Вариант A (установить зависимости):**
```bash
pip install -r requirements.txt
```

**Вариант B (установить конкретный пакет):**
```bash
pip install stable_baselines3 gymnasium pandas matplotlib imageio
```

**Вариант C (проверьте окружение):**
```bash
# Проверьте активное окружение
conda info --envs
# Активируйте правильное
conda activate rocm

# Проверьте установленные пакеты
pip list | grep stable_baselines3
```

---

### 3. Проблема: Память не хватает при генерации видео

**Симптомы:**
```bash
Killed
MemoryError
```

**Решения:**

**Вариант A (меньше эпизодов):**
```bash
python -m src.visualization.video \
    --model results/experiments/ppo_seed42/ppo_seed42_model.zip \
    --output video.mp4 \
    --episodes 2  # Было 5
```

**Вариант B (меньше FPS):**
```bash
python -m src.visualization.video \
    --model results/experiments/ppo_seed42/ppo_seed42_model.zip \
    --output video.mp4 \
    --episodes 5 \
    --fps 15  # Было 30
```

**Вариант C (освободите память):**
```bash
# Проверьте свободную память
free -h

# Закройте другие процессы
killall python
```

---

## 🎓 Проблемы с обучением

### 4. Проблема: Обучение идёт очень медленно

**Симптомы:**
- Обучение 200K шагов занимает > 1 часа
- CPU загружен на 100%

**Решения:**

**Вариант A (проверьте seed):**
```bash
# Используйте стандартный seed
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 200000 --seed 42
```

**Вариант B (меньше шагов для тестирования):**
```bash
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 50000 --seed 42
```

**Вариант C (проверьте окружение):**
```bash
# LunarLander-v3 может быть медленным на некоторых системах
# Используйте LunarLander-v2 вместо v3
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 200000 --seed 42 --env-id LunarLander-v2
```

---

### 5. Проблема: Reward не растёт / модель не сходится

**Симптомы:**
- Reward остаётся отрицательным
- Reward сильно колеблется
- Через 200K шагов reward < 100

**Решения:**

**Вариант A (больше шагов):**
```bash
# Увеличьте количество шагов до 500K-1M
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 500000 --seed 42
```

**Вариант B (проверьте gamma):**
```bash
# Попробуйте разные gamma значения
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 200000 --seed 42 --gamma 0.99
```

**Вариант C (другой seed):**
```bash
# Попробуйте другой seed
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 200000 --seed 123
```

**Вариант D (проверьте learning rate):**
- Измените learning rate в конфиге
- Значение по умолчанию: 3e-4
- Попробуйте: 1e-4, 5e-4

---

### 6. Проблема: Модель не загружается

**Симптомы:**
```python
from stable_baselines3 import PPO
model = PPO.load("results/experiments/ppo_seed42/ppo_seed42_model.zip")
# FileNotFoundError или коррупция файла
```

**Решения:**

**Вариант A (проверьте путь):**
```bash
# Убедитесь, что путь правильный
ls -la results/experiments/ppo_seed42/
```

**Вариант B (проверьте целостность):**
```bash
# Проверьте, что это валидный ZIP файл
unzip -l results/experiments/ppo_seed42/ppo_seed42_model.zip
```

**Вариант C (переобучите модель):**
```bash
python -m src.experiments.completion.baseline_training \
    --algo ppo --timesteps 200000 --seed 42
```

---

## 📊 Проблемы с визуализацией

### 7. Проблема: График не генерируется

**Симптомы:**
```bash
python -m src.visualization.graphs \
    --experiment ppo_seed42 --type learning_curve
# FileNotFoundError или нет файла metrics.csv
```

**Решения:**

**Вариант A (проверьте metrics.csv):**
```bash
# Убедитесь, что файл существует и не пустой
cat results/experiments/ppo_seed42/metrics.csv
wc -l results/experiments/ppo_seed42/metrics.csv
```

**Вариант B (проверьте эксперимент ID):**
```bash
# Список всех экспериментов
python scripts/results_summarizer.py --list
```

**Вариант C (проверьте структуру CSV):**
```bash
# Проверьте колонки
head -n 1 results/experiments/ppo_seed42/metrics.csv
# Должно быть: timesteps,reward_mean,reward_std,...
```

---

### 8. Проблема: Видео не воспроизводится

**Симптомы:**
- Видео создаётся, но не открывается
- Видео чёрное или зелёное
- Видеоплеер не поддерживает кодек

**Решения:**

**Вариант A (проверьте файл):**
```bash
# Проверьте размер файла
ls -lh results/experiments/ppo_seed42/video.mp4
# Должно быть > 50KB

# Проверьте метаданные
ffmpeg -i results/experiments/ppo_seed42/video.mp4
```

**Вариант B (используйте другой кодек):**
```bash
# Установите imageio-ffmpeg
pip install imageio[ffmpeg]
```

**Вариант C (пересоздайте видео):**
```bash
python -m src.visualization.video \
    --model results/experiments/ppo_seed42/ppo_seed42_model.zip \
    --output video.mp4 --episodes 3 --fps 24
```

---

## 🧪 Проблемы с тестами

### 9. Проблема: Тесты не проходят

**Симптомы:**
```bash
pytest tests/unit/ -v
# FAILED ...
```

**Решения:**

**Вариант A (запустите конкретный тест):**
```bash
# Запустите конкретный тест
pytest tests/unit/test_callbacks.py::test_checkpoint_callback -v

# Запустите с выводом
pytest tests/unit/test_callbacks.py::test_checkpoint_callback -vv
```

**Вариант B (обновите зависимости):**
```bash
pip install -r requirements.txt --upgrade
```

**Вариант C (очистите кэш):**
```bash
# Очистите pytest кэш
rm -rf .pytest_cache __pycache__
# Пересоздайте
pytest tests/unit/ -v
```

---

## ✅ Проблемы с качеством кода

### 10. Проблема: `ruff check` выдаёт ошибки

**Симптомы:**
```bash
ruff check .
# F401, F841, E501, ...
```

**Решения:**

**Вариант A (автоисправление):**
```bash
# Автоисправить ошибки
ruff check . --fix
```

**Вариант B (проверьте конкретный файл):**
```bash
ruff check src/training/trainer.py
ruff check src/training/trainer.py --fix
```

**Вариант C (игнорируйте конкретные ошибки):**
```bash
# Создайте .ruff.toml
echo "[lint]
ignore = ['F401']" > .ruff.toml
```

### 11. Проблема: `mypy` выдаёт ошибки

**Симптомы:**
```bash
mypy src/ --strict
# error: ...
```

**Решения:**

**Вариант A (проверьте конкретный файл):**
```bash
mypy src/training/trainer.py --strict
```

**Вариант B (используйте менее строгий режим):**
```bash
mypy src/  # Без --strict
```

**Вариант C (игнорируйте конкретные ошибки):**
```bash
# Создайте mypy.ini
echo "[mypy]
ignore_missing_imports = True" > mypy.ini
```

---

## 📞 Дополнительная помощь

Если проблема не решена:

1. **Проверьте документацию:**
   - [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
   - [КОМАНДЫ.md](КОМАНДЫ.md)
   - [QUICKSTART.md](docs/QUICKSTART.md)

2. **Проверьте логи:**
   - `results/logs/` - логи обучения
   - `results/experiments/*/metrics.csv` - метрики

3. **Запустите диагностику:**
   ```bash
   # Проверьте артефакты
   python scripts/verify_artifacts.py --all

   # Проверьте зависимости
   python run_experiments.py --check-deps
   ```

4. **Сообщите о проблеме:**
   - Скопируйте traceback
   - Включите команду, которую вы запускали
   - Приложите конфигурацию (config.json)

---

## 🔧 Полная диагностика

Запустите полную диагностику:

```bash
#!/bin/bash

echo "=== DIAGNOSTIC SCRIPT ==="

echo ""
echo "[1] Python version:"
python --version

echo ""
echo "[2] Conda environment:"
conda info --envs

echo ""
echo "[3] Installed packages:"
pip list | grep -E "(stable-baselines3|gymnasium|torch)"

echo ""
echo "[4] Disk space:"
df -h

echo ""
echo "[5] Memory:"
free -h

echo ""
echo "[6] Verify artifacts:"
python scripts/verify_artifacts.py --all

echo ""
echo "[7] Run unit tests:"
pytest tests/unit/ -v --tb=short

echo ""
echo "[8] Lint check:"
ruff check . --statistics

echo ""
echo "[9] Type check:"
mypy src/ --strict 2>&1 | head -20

echo ""
echo "=== END OF DIAGNOSTIC ==="
```

Сохраните как `diagnose.sh` и запустите:
```bash
chmod +x diagnose.sh
./diagnose.sh
```

---

*Последнее обновление: 2026-02-03*
