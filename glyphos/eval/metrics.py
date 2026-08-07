"""Translation metrics (§ Phase 4.1) — chrF primary, BLEU secondary.

COMET is excluded from headline metrics by design: a learned metric IS a
pretrained model (hard constraint §1), and it is known to be insensitive to
exactly the contextual distinctions this project cares about.

sacrebleu is a metric *implementation*, not a model — no weights are loaded.
Signatures are recorded with every score so numbers are reproducible and
comparable to published work.
"""

from dataclasses import dataclass, field

import sacrebleu


@dataclass(frozen=True)
class TranslationScores:
    chrf: float  # primary
    bleu: float  # secondary
    n: int
    chrf_signature: str = ""
    bleu_signature: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "chrf": self.chrf,
            "bleu": self.bleu,
            "n": self.n,
            "chrf_signature": self.chrf_signature,
            "bleu_signature": self.bleu_signature,
            **self.extra,
        }


def score_translations(
    hypotheses: list[str], references: list[str], word_order: int = 0
) -> TranslationScores:
    """chrF (word_order=2 gives chrF++) and BLEU over a parallel corpus."""
    if len(hypotheses) != len(references):
        raise ValueError(
            f"hypotheses ({len(hypotheses)}) and references ({len(references)}) differ in length"
        )
    if not hypotheses:
        raise ValueError("empty hypothesis list")

    chrf_metric = sacrebleu.CHRF(word_order=word_order)
    bleu_metric = sacrebleu.BLEU()
    chrf = chrf_metric.corpus_score(hypotheses, [references])
    bleu = bleu_metric.corpus_score(hypotheses, [references])
    return TranslationScores(
        chrf=chrf.score,
        bleu=bleu.score,
        n=len(hypotheses),
        chrf_signature=str(chrf_metric.get_signature()),
        bleu_signature=str(bleu_metric.get_signature()),
    )


def sentence_chrf(hypothesis: str, reference: str, word_order: int = 0) -> float:
    """Per-sentence chrF — the unit the document-level bootstrap resamples."""
    return sacrebleu.CHRF(word_order=word_order).sentence_score(hypothesis, [reference]).score
