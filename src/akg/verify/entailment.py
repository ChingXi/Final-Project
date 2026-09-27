
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

CONTRADICTION_THRESHOLD = 0.9

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"


@dataclass(frozen=True)
class Entailment:

    label: str          # "entailment" | "neutral" | "contradiction"
    probability: float  # of `label`

    @property
    def contradicted(self) -> bool:
        """Is this a confident contradiction, i.e. worth acting on?"""
        return (
            self.label == "contradiction"
            and self.probability >= CONTRADICTION_THRESHOLD
        )


@runtime_checkable
class EntailmentChecker(Protocol):

    def check(self, pairs: list[tuple[str, str]]) -> list[Entailment]: ...


class CrossEncoderChecker:

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._labels: dict[int, str] = {}

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            self._labels = {
                int(i): str(name).lower()
                for i, name in self._model.config.id2label.items()
            }
        return self._model

    def check(self, pairs: list[tuple[str, str]]) -> list[Entailment]:
        if not pairs:
            return []
        model = self._load()
        scored = model.predict(pairs, apply_softmax=True, show_progress_bar=False)
        out: list[Entailment] = []
        for row in scored:
            best = max(range(len(row)), key=lambda i: row[i])
            out.append(
                Entailment(
                    label=self._labels.get(best, str(best)),
                    probability=round(float(row[best]), 3),
                )
            )
        return out
