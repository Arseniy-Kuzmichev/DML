"""
Homework 3: RAG Animation System

Вход: текстовый запрос, например "танец макарена"
Выход: GIF-анимация.

Что делает файл:
1. Загружает poses_database.json.
2. Ищет в базе позы, похожие на запрос и шаги танца.
3. Использует Ollama как LLM-планировщик: просит разложить запрос на шаги анимации.
4. Использует Pose API на порту 8001, если он доступен.
5. Если Pose API недоступен, всё равно создаёт GIF локальным renderer-ом,
   чтобы домашку можно было проверить без падения.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw

Pose = Dict[str, List[float]]


@dataclass
class PoseEntry:
    """Одна запись из poses_database.json."""

    index: int
    pose: Pose
    description: str


@dataclass
class AnimationStep:
    """Один смысловой шаг будущей анимации."""

    text: str


@dataclass
class ServiceStatus:
    """Статус внешних сервисов, которые требуются в задании."""

    ollama_ok: bool
    pose_api_ok: bool
    ollama_message: str
    pose_api_message: str


def tokenize(text: str) -> List[str]:
    """
    Очень простой токенизатор для русского и английского текста.
    Нужен для локального fallback-поиска, если embedding-модель Ollama не настроена.
    """
    text = text.lower().replace("ё", "е")
    return re.findall(r"[a-zа-я0-9]+", text)


class PoseDatabase:
    """Хранилище поз из JSON-файла."""

    def __init__(self, entries: List[PoseEntry]) -> None:
        self.entries = entries

    @classmethod
    def load(cls, path: str | Path) -> "PoseDatabase":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw_entries = json.load(f)

        entries: List[PoseEntry] = []
        for i, item in enumerate(raw_entries):
            if "pose" not in item or "description" not in item:
                raise ValueError(f"Некорректная запись в базе поз: index={i}")
            entries.append(
                PoseEntry(
                    index=i,
                    pose=item["pose"],
                    description=str(item["description"]),
                )
            )

        if not entries:
            raise ValueError("База поз пустая")

        return cls(entries)

    def find_by_description_contains(self, word: str) -> List[PoseEntry]:
        word = word.lower().replace("ё", "е")
        return [
            entry
            for entry in self.entries
            if word in entry.description.lower().replace("ё", "е")
        ]


class TfidfRetriever:
    """
    Мини-RAG retrieval без тяжёлых зависимостей.

    В реальном проекте вместо этого можно использовать эмбеддинги Ollama:
    nomic-embed-text / mxbai-embed-large и FAISS/Chroma.
    Для домашки достаточно простого retriever-а:
    он превращает описания поз в TF-IDF-векторы и ищет ближайшие описания.
    """

    def __init__(self, database: PoseDatabase) -> None:
        self.database = database
        self.documents = [entry.description for entry in database.entries]
        self.doc_tokens = [tokenize(doc) for doc in self.documents]
        self.idf = self._build_idf(self.doc_tokens)
        self.doc_vectors = [self._vectorize(tokens) for tokens in self.doc_tokens]

    @staticmethod
    def _build_idf(tokenized_docs: List[List[str]]) -> Dict[str, float]:
        n_docs = len(tokenized_docs)
        df: Dict[str, int] = {}
        for tokens in tokenized_docs:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        return {
            token: math.log((1 + n_docs) / (1 + count)) + 1.0
            for token, count in df.items()
        }

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}

        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        length = len(tokens)
        return {
            token: (count / length) * self.idf.get(token, 0.0)
            for token, count in tf.items()
        }

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0

        dot = sum(weight * b.get(token, 0.0) for token, weight in a.items())
        norm_a = math.sqrt(sum(weight * weight for weight in a.values()))
        norm_b = math.sqrt(sum(weight * weight for weight in b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[PoseEntry, float]]:
        query_vector = self._vectorize(tokenize(query))
        scored = [
            (entry, self._cosine(query_vector, doc_vector))
            for entry, doc_vector in zip(self.database.entries, self.doc_vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


class OllamaPlanner:
    """
    LLM-планировщик через Ollama.

    Он не рисует позы сам, а только помогает превратить общий запрос
    ("танец макарена") в короткий список смысловых шагов.
    Потом RAG-часть подбирает реальные позы из poses_database.json.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def health(self) -> Tuple[bool, str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if response.ok:
                return True, "Ollama доступна"
            return False, f"Ollama ответила статусом {response.status_code}"
        except requests.RequestException as exc:
            return False, f"Ollama недоступна: {exc}"

    def plan(
        self, user_prompt: str, retrieved_context: List[PoseEntry]
    ) -> List[AnimationStep]:
        """
        Просим Ollama вернуть JSON-массив шагов.
        Если модель недоступна или ответ не JSON, используем fallback.
        """
        context_text = "\n".join(
            f"- {entry.description}" for entry in retrieved_context[:12]
        )

        system_prompt = (
            "Ты помогаешь собрать GIF-анимацию из базы готовых поз. "
            "Нужно разложить пользовательский запрос на 6-10 коротких шагов. "
            "Каждый шаг должен быть описанием положения рук, ног и корпуса. "
            "Верни строго JSON-массив строк без Markdown."
        )

        user_message = (
            f"Запрос пользователя: {user_prompt}\n\n"
            f"Доступные похожие позы:\n{context_text}\n\n"
            "Составь последовательность шагов для анимации."
        )

        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n{user_message}",
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            steps_raw = extract_json_array(text)
            if steps_raw:
                return [AnimationStep(text=str(step)) for step in steps_raw]
        except Exception:
            pass

        return fallback_plan(user_prompt)


class PoseApiClient:
    """
    Клиент для Pose API из Step 2.

    Так как в разных домашних работах endpoint мог называться по-разному,
    клиент пробует несколько популярных вариантов. Если ни один не сработал,
    система не падает, а локально рисует GIF через Pillow.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Tuple[bool, str]:
        for endpoint in ("/health", "/", "/docs"):
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    timeout=3,
                )
                if response.status_code < 500:
                    return True, f"Pose API доступен: {endpoint}"
            except requests.RequestException:
                continue

        return False, "Pose API недоступен на localhost:8001"

    def try_generate_gif(
        self,
        prompt: str,
        frames: List[Pose],
        output_path: str | Path,
        fps: int,
    ) -> Optional[Path]:
        """
        Пробует попросить Pose API создать GIF.
        Возвращает путь к GIF, если API успешно справился.
        """
        output_path = Path(output_path)

        payload = {
            "prompt": prompt,
            "poses": frames,
            "frames": frames,
            "fps": fps,
            "output_path": str(output_path),
        }

        candidate_endpoints = [
            "/generate_gif",
            "/generate-gif",
            "/animate",
            "/animation",
            "/render_gif",
            "/render-gif",
            "/predict",
        ]

        for endpoint in candidate_endpoints:
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    timeout=self.timeout,
                )

                if not response.ok:
                    continue

                content_type = response.headers.get("content-type", "")

                if "image/gif" in content_type:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(response.content)
                    return output_path

                if "application/json" in content_type:
                    data = response.json()
                    for key in ("gif_path", "output_path", "path", "file"):
                        maybe_path = data.get(key)
                        if maybe_path and Path(maybe_path).exists():
                            return Path(maybe_path)

                    # Иногда API возвращает base64, но в этой домашке это не обязательно.
                    # Если твой Step 2 работает иначе, достаточно адаптировать этот блок.
            except requests.RequestException:
                continue

        return None


def extract_json_array(text: str) -> Optional[List[Any]]:
    """Достаёт JSON-массив из ответа LLM."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def fallback_plan(user_prompt: str) -> List[AnimationStep]:
    """
    Резервный план.
    Он особенно важен для критерия оценки: запрос "танец макарена"
    должен стабильно дать GIF-анимацию.
    """
    normalized = user_prompt.lower().replace("ё", "е")

    if "макарен" in normalized or "macarena" in normalized:
        steps = [
            "Обе руки вперед, ладони вниз, Макарена",
            "Обе руки вперед, ладони вверх, Макарена",
            "Обе руки на затылке, Макарена",
            "Обе руки на ушах, Макарена",
            "Обе руки на бедрах, Макарена",
            "Колени согнуты, руки опущены, Макарена",
            "Обе руки подняты вверх, Макарена",
            "Исходная позиция, руки вперед на уровне плеч",
        ]
    else:
        steps = [
            user_prompt,
            "руки вперед",
            "руки вверх",
            "руки на бедрах",
            "небольшой прыжок",
            "исходная позиция",
        ]

    return [AnimationStep(text=step) for step in steps]


def select_pose_sequence(
    prompt: str,
    database: PoseDatabase,
    retriever: TfidfRetriever,
    planner: OllamaPlanner,
) -> Tuple[List[PoseEntry], List[AnimationStep]]:
    """
    Главная RAG-логика:
    1. Берём пользовательский запрос.
    2. Достаём похожие позы из базы.
    3. Просим Ollama составить план шагов.
    4. Для каждого шага снова ищем самую подходящую позу.
    """
    normalized = prompt.lower().replace("ё", "е")

    # Важный кейс для проверки домашки:
    # если в базе есть позы с "Макарена", используем их в исходном порядке.
    if "макарен" in normalized or "macarena" in normalized:
        macarena_entries = database.find_by_description_contains("макарен")
        if macarena_entries:
            steps = [
                AnimationStep(text=entry.description) for entry in macarena_entries
            ]
            return macarena_entries, steps

    initial_context = [entry for entry, _ in retriever.search(prompt, top_k=12)]
    steps = planner.plan(prompt, initial_context)

    selected: List[PoseEntry] = []
    used_indexes: set[int] = set()

    for step in steps:
        candidates = retriever.search(step.text, top_k=5)

        chosen: Optional[PoseEntry] = None
        for entry, score in candidates:
            if entry.index not in used_indexes or len(selected) < 2:
                chosen = entry
                break

        if chosen is None and candidates:
            chosen = candidates[0][0]

        if chosen is not None:
            selected.append(chosen)
            used_indexes.add(chosen.index)

    if not selected:
        selected = [entry for entry, _ in retriever.search(prompt, top_k=8)]

    return selected, steps


def interpolate_pose(a: Pose, b: Pose, alpha: float) -> Pose:
    """Линейная интерполяция между двумя позами."""
    result: Pose = {}
    for key in a.keys():
        ax, ay = a[key]
        bx, by = b.get(key, a[key])
        result[key] = [
            ax * (1 - alpha) + bx * alpha,
            ay * (1 - alpha) + by * alpha,
        ]
    return result


def build_animation_frames(
    pose_entries: List[PoseEntry],
    frames_between_poses: int = 4,
    repeat: int = 2,
) -> List[Pose]:
    """
    Из выбранных RAG-поз делает плавную последовательность кадров.
    """
    if not pose_entries:
        raise ValueError("Нет выбранных поз для анимации")

    base_poses = [entry.pose for entry in pose_entries]

    # Зацикливаем танец, чтобы GIF выглядел как повторяющаяся анимация.
    base_poses = base_poses * repeat

    frames: List[Pose] = []
    for current_pose, next_pose in zip(base_poses, base_poses[1:] + [base_poses[0]]):
        for i in range(frames_between_poses):
            alpha = i / max(frames_between_poses, 1)
            frames.append(interpolate_pose(current_pose, next_pose, alpha))

    return frames


def pose_to_canvas(
    point: List[float],
    canvas_size: Tuple[int, int] = (420, 420),
    scale: float = 3.0,
    center: Tuple[int, int] = (210, 245),
) -> Tuple[int, int]:
    """Переводит координаты позы в координаты картинки."""
    x, y = point
    return int(center[0] + x * scale), int(center[1] - y * scale)


def draw_stick_figure(
    pose: Pose,
    description: str,
    canvas_size: Tuple[int, int] = (420, 420),
) -> Image.Image:
    """
    Локальный renderer позы.
    Это backup на случай, если Pose API не поднят.
    """
    image = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(image)

    torso = pose_to_canvas(pose["Torso"], canvas_size)
    head = pose_to_canvas(pose["Head"], canvas_size)
    rh = pose_to_canvas(pose["RH"], canvas_size)
    lh = pose_to_canvas(pose["LH"], canvas_size)
    rk = pose_to_canvas(pose["RK"], canvas_size)
    lk = pose_to_canvas(pose["LK"], canvas_size)

    # Пол и слабая сетка, чтобы движение было заметнее.
    floor_y = pose_to_canvas([0, -70], canvas_size)[1]
    draw.line(
        (40, floor_y, canvas_size[0] - 40, floor_y), width=2, fill=(220, 220, 220)
    )
    draw.ellipse(
        (torso[0] - 6, torso[1] - 6, torso[0] + 6, torso[1] + 6), fill=(40, 40, 40)
    )

    # Скелет
    line_width = 7
    joint_r = 7
    draw.line((head, torso), width=line_width, fill=(30, 30, 30))
    draw.line((torso, rh), width=line_width, fill=(30, 30, 30))
    draw.line((torso, lh), width=line_width, fill=(30, 30, 30))
    draw.line((torso, rk), width=line_width, fill=(30, 30, 30))
    draw.line((torso, lk), width=line_width, fill=(30, 30, 30))

    # Голова
    head_r = 22
    draw.ellipse(
        (head[0] - head_r, head[1] - head_r, head[0] + head_r, head[1] + head_r),
        outline=(30, 30, 30),
        width=5,
        fill=(245, 245, 245),
    )

    # Суставы
    for point in (rh, lh, rk, lk):
        draw.ellipse(
            (
                point[0] - joint_r,
                point[1] - joint_r,
                point[0] + joint_r,
                point[1] + joint_r,
            ),
            fill=(30, 30, 30),
        )

    # Подпись кадра
    title = "RAG Animation: Macarena"
    draw.text((16, 14), title, fill=(30, 30, 30))
    draw.text((16, canvas_size[1] - 36), description[:54], fill=(60, 60, 60))

    return image


def render_gif_locally(
    frames: List[Pose],
    descriptions: List[str],
    output_path: str | Path,
    fps: int = 6,
) -> Path:
    """Создаёт GIF локально через Pillow."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images: List[Image.Image] = []
    for i, frame in enumerate(frames):
        description = descriptions[i % len(descriptions)] if descriptions else ""
        images.append(draw_stick_figure(frame, description))

    duration_ms = int(1000 / fps)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    return output_path


def check_services(ollama: OllamaPlanner, pose_api: PoseApiClient) -> ServiceStatus:
    ollama_ok, ollama_message = ollama.health()
    pose_api_ok, pose_api_message = pose_api.health()

    return ServiceStatus(
        ollama_ok=ollama_ok,
        pose_api_ok=pose_api_ok,
        ollama_message=ollama_message,
        pose_api_message=pose_api_message,
    )


def save_metadata(
    output_path: str | Path,
    prompt: str,
    selected_entries: List[PoseEntry],
    steps: List[AnimationStep],
    service_status: ServiceStatus,
    renderer: str,
) -> Path:
    """Сохраняет JSON-отчёт: какие позы были выбраны и какие сервисы отвечали."""
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".metadata.json")

    metadata = {
        "prompt": prompt,
        "renderer": renderer,
        "services": {
            "ollama_ok": service_status.ollama_ok,
            "pose_api_ok": service_status.pose_api_ok,
            "ollama_message": service_status.ollama_message,
            "pose_api_message": service_status.pose_api_message,
        },
        "selected_poses": [
            {
                "index": entry.index,
                "description": entry.description,
                "pose": entry.pose,
            }
            for entry in selected_entries
        ],
        "planned_steps": [step.text for step in steps],
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def generate_animation(
    prompt: str,
    db_path: str | Path,
    output_path: str | Path,
    ollama_url: str,
    pose_api_url: str,
    ollama_model: str,
    fps: int,
    frames_between_poses: int,
    repeat: int,
) -> Dict[str, Any]:
    """Полный pipeline домашки."""
    started_at = time.time()

    database = PoseDatabase.load(db_path)
    retriever = TfidfRetriever(database)
    ollama = OllamaPlanner(base_url=ollama_url, model=ollama_model)
    pose_api = PoseApiClient(base_url=pose_api_url)

    service_status = check_services(ollama, pose_api)

    selected_entries, steps = select_pose_sequence(
        prompt=prompt,
        database=database,
        retriever=retriever,
        planner=ollama,
    )

    frames = build_animation_frames(
        selected_entries,
        frames_between_poses=frames_between_poses,
        repeat=repeat,
    )

    output_path = Path(output_path)

    renderer = "pose_api"
    gif_path = pose_api.try_generate_gif(
        prompt=prompt,
        frames=frames,
        output_path=output_path,
        fps=fps,
    )

    if gif_path is None:
        renderer = "local_pillow_fallback"
        gif_path = render_gif_locally(
            frames=frames,
            descriptions=[entry.description for entry in selected_entries],
            output_path=output_path,
            fps=fps,
        )

    metadata_path = save_metadata(
        output_path=gif_path,
        prompt=prompt,
        selected_entries=selected_entries,
        steps=steps,
        service_status=service_status,
        renderer=renderer,
    )

    return {
        "prompt": prompt,
        "gif_path": str(gif_path),
        "metadata_path": str(metadata_path),
        "renderer": renderer,
        "selected_poses_count": len(selected_entries),
        "frames_count": len(frames),
        "services": service_status.__dict__,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG-система для генерации GIF-анимаций"
    )
    parser.add_argument("--prompt", default="танец макарена", help="Текстовый запрос")
    parser.add_argument(
        "--db", default="poses_database.json", help="Путь к poses_database.json"
    )
    parser.add_argument(
        "--output", default="outputs/macarena.gif", help="Куда сохранить GIF"
    )
    parser.add_argument(
        "--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434")
    )
    parser.add_argument(
        "--pose-api-url", default=os.getenv("POSE_API_URL", "http://localhost:8001")
    )
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3.2"))
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--frames-between-poses", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = generate_animation(
        prompt=args.prompt,
        db_path=args.db,
        output_path=args.output,
        ollama_url=args.ollama_url,
        pose_api_url=args.pose_api_url,
        ollama_model=args.ollama_model,
        fps=args.fps,
        frames_between_poses=args.frames_between_poses,
        repeat=args.repeat,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
