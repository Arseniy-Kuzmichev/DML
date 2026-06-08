# Step 4: FastAPI Monitoring

Эта домашка реализует мониторинг FastAPI-сервиса инференса ONNX-модели из шага 2.
Мониторинг проверяет `/health` и `/predict`, считает latency/error-rate, пишет JSON-логи и показывает цветные алерты в консоли.

## Что есть в проекте

```text
step4_monitoring/
├── main.py
├── src/
│   ├── monitor.py
│   ├── logger.py
│   └── config.py
├── config/
│   └── monitoring_config.yaml
├── logs/
├── test_images/
├── requirements.txt
└── README.md
```

## Что мониторится

- `Response Time` — среднее время ответа по запросам в одном цикле проверки.
- `P95 Latency` — 95-й перцентиль времени ответа.
- `Error Rate` — процент неудачных запросов.
- `Health Status` — результат проверки `/health`.
- `Consecutive Failures` — количество последовательных ошибок.

## Цвета алертов

- Зеленый — нормальная работа.
- Желтый — превышены warning-пороги.
- Красный — превышены critical-пороги или `/health` не отвечает корректно.

## Установка

Из папки `step4_monitoring`:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Настройка

Откройте `config/monitoring_config.yaml`.

Если FastAPI-сервис из шага 2 уже запущен на `http://localhost:8000`, ничего менять не нужно:

```yaml
service:
  base_url: "http://localhost:8000"
  start_command: ""
```

Если хотите, чтобы мониторинг сам запускал сервис, заполните `start_command`:

```yaml
service:
  start_command: "uvicorn app.main:app --host 0.0.0.0 --port 8000"
```

Команду нужно заменить на ту, которой запускается ваш FastAPI-сервис из шага 2.

## Тестовые изображения

Для проверки `/predict` положите любые изображения в папку:

```text
test_images/
```

Поддерживаются форматы: `jpg`, `jpeg`, `png`, `bmp`, `webp`, `tif`, `tiff`.

Если папка пустая, мониторинг отправит маленькую тестовую PNG-картинку. Но для реальной проверки модели лучше положить обычное изображение, похожее на входные данные вашей модели.

## Запуск

Один цикл проверки:

```bash
python main.py --once
```

Постоянный мониторинг:

```bash
python main.py
```

По умолчанию проверка запускается каждые 30 секунд.

## Где смотреть результат

Консоль покажет цветной статус:

```text
[2026-06-07 18:00:00] INFO     [NORMAL] health=True | avg=124.5ms | p95=180.1ms | errors=0.0% | consecutive_failures=0
```

Файл структурированных логов:

```text
logs/monitoring.log
```

Файл метрик в формате JSONL:

```text
logs/metrics.jsonl
```

Каждая строка в `metrics.jsonl` — отдельный JSON-объект с метриками одного цикла проверки.

## Как изменить пороги

Пороги задаются в `config/monitoring_config.yaml`:

```yaml
thresholds:
  response_time_ms:
    warning: 2000
    critical: 5000
  p95_latency_ms:
    warning: 3000
    critical: 6000
  error_rate_percent:
    warning: 10
    critical: 25
  consecutive_failures:
    warning: 3
    critical: 5
```

Например, если хотите быстрее получать предупреждение по latency, уменьшите `warning`.

## Важный момент про `/predict`

В коде по умолчанию считается, что endpoint принимает файл в поле `file`, то есть примерно так:

```python
file: UploadFile = File(...)
```

Если в вашем FastAPI-сервисе поле называется иначе, поменяйте это в конфиге:

```yaml
endpoints:
  predict_file_field: "image"
```

Если API должен возвращать конкретные поля, их можно перечислить здесь:

```yaml
endpoints:
  expected_prediction_keys: ["class", "confidence"]
```

Если список пустой, мониторинг принимает любой валидный JSON-ответ.
