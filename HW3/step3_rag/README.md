# Homework 3: RAG Animation System

## Задание

Нужно реализовать RAG-систему, которая по текстовому запросу создаёт GIF-анимацию.

Пример:

```text
Вход: танец макарена
Выход: outputs/macarena.gif
```

В проекте используется база `poses_database.json`, где лежат 96 поз с координатами частей тела и текстовыми описаниями.

## Что такое RAG в этой домашке

RAG здесь работает так:

1. Пользователь вводит текстовый запрос, например `танец макарена`.
2. Система ищет в `poses_database.json` позы, похожие на этот запрос.
3. Ollama используется как LLM-планировщик: она помогает разложить запрос на шаги танца.
4. Для каждого шага система снова ищет наиболее подходящую позу в базе.
5. Из найденных поз собирается последовательность кадров.
6. Pose API используется для генерации GIF, если сервис доступен.
7. Если Pose API не поднят, включается локальный fallback renderer на Pillow, чтобы GIF всё равно создался.

## Структура проекта

```text
rag_animation_homework/
├── rag_animation.py          # основной код RAG-системы
├── poses_database.json       # база поз
├── requirements.txt          # зависимости
├── run_macarena.sh           # быстрый запуск
└── outputs/
    ├── macarena.gif          # результат генерации
    └── macarena.metadata.json
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

## Перед запуском

По условию используются сервисы из предыдущих шагов:

- Ollama: `http://localhost:11434`
- Pose API: `http://localhost:8001`

Проверьте, что они запущены.

Пример для Ollama:

```bash
ollama serve
```

Если у вас модель называется не `llama3.2`, можно передать её через переменную:

```bash
export OLLAMA_MODEL=llama3
```

или через аргумент:

```bash
python rag_animation.py --ollama-model llama3
```

## Запуск

```bash
python rag_animation.py --prompt "танец макарена" --output outputs/macarena.gif
```

Или так:

```bash
bash run_macarena.sh
```

## Что должно получиться на выходе

После запуска появятся файлы:

```text
outputs/macarena.gif
outputs/macarena.metadata.json
```

`macarena.gif` — сама GIF-анимация.

`macarena.metadata.json` — отчёт, где видно:

- какой запрос был обработан;
- какие позы выбрала RAG-система;
- был ли доступен Ollama;
- был ли доступен Pose API;
- каким renderer-ом создан GIF.

## Почему запрос «танец макарена» точно работает

В базе есть позы, в описании которых прямо указано `Макарена`.
Для этого запроса система сначала достаёт именно эти позы и собирает из них танцевальную последовательность.

## Если Pose API не работает

Скрипт не падает. Он пишет в metadata:

```json
"renderer": "local_pillow_fallback"
```

Это значит, что GIF создан локально через Pillow.

Такой fallback нужен, чтобы домашку можно было проверить даже без запущенного Step 2.
При этом интеграция с Pose API всё равно есть: код проверяет сервис на `localhost:8001` и пробует отправить туда кадры.

## Основная логика кода

Главные классы и функции:

- `PoseDatabase` — загружает базу поз.
- `TfidfRetriever` — ищет похожие позы по описаниям.
- `OllamaPlanner` — обращается к Ollama и строит план анимации.
- `PoseApiClient` — обращается к Pose API.
- `select_pose_sequence()` — выбирает последовательность поз.
- `build_animation_frames()` — делает плавные промежуточные кадры.
- `render_gif_locally()` — запасная локальная генерация GIF.
- `generate_animation()` — полный pipeline.

## Команда для проверки критерия оценки

```bash
python rag_animation.py --prompt "танец макарена"
```

Критерий выполнен, если после команды появился файл:

```text
outputs/macarena.gif
```
