[![CI](https://github.com/<Spamik228>/<Lab-04>/actions/workflows/ci.yml/badge.svg)](https://github.com/<Spamik228>/<Lab-04>/actions/workflows/ci.yml)
## Встановлення та запуск

### 1. Встановлення як глобальну утиліту (CLI)

Завдяки `uv` інструмент можна встановити безпосередньо в систему або у віртуальне середовище:

```bash
# Встановлення з поточного вихідного коду
uv tool install .

# Або встановлення зі зібраного wheel-файлу
uv build
uv tool install dist/findex4-0.1.0-py3-none-any.whl