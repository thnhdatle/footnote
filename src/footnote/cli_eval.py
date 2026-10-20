# SPDX-License-Identifier: Apache-2.0
"""Evaluation CLI group: `eval retrieval|abstention|calibrate|answers|cache status`."""

from __future__ import annotations

import asyncio
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from footnote.cli_common import EXIT_FAILURE, fail, settings_from
from footnote.config import EXIT_CODE_INVALID

if TYPE_CHECKING:
    from footnote.app.evaluation.abstention import Calibration, GateSpec
    from footnote.app.evaluation.answers import AnswerEvalResult
    from footnote.app.evaluation.report import Row
    from footnote.app.evaluation.retrieval import EvalOutcome, Variant
    from footnote.app.evaluation.runtime import EvalRuntime
    from footnote.config import Settings

eval_app = typer.Typer(help="Evaluation harness.", no_args_is_help=True)
cache_app = typer.Typer(help="Embedding cache of the evaluation harness.", no_args_is_help=True)
eval_app.add_typer(cache_app, name="cache")

DEFAULT_DATASETS = "scifact,trec-covid"


@cache_app.command("status")
def cache_status(
    ctx: typer.Context,
    require: Annotated[str, typer.Option(help="Comma-separated datasets that must be warm.")] = "",
) -> None:
    """Print warm/cold and data present/absent per dataset; exit 1 unless every required
    dataset is warm."""
    from footnote.app.evaluation.cache import is_warm  # noqa: PLC0415
    from footnote.app.evaluation.datasets import dataset_dir  # noqa: PLC0415
    from footnote.config import ConfigError  # noqa: PLC0415

    try:
        settings = settings_from(ctx)
    except ConfigError as exc:
        raise fail(str(exc), EXIT_CODE_INVALID) from exc
    required = [n.strip() for n in require.split(",") if n.strip()]
    names = list(dict.fromkeys([*DEFAULT_DATASETS.split(","), *required]))
    cold: list[str] = []
    for name in names:
        try:
            dataset_dir(name)
            data = "present"
        except FileNotFoundError:
            data = "absent"
        warm = is_warm(settings.eval.cache_dir, name)
        if name in required and not warm:
            cold.append(name)
        typer.echo(f"{name}\t{'warm' if warm else 'cold'}\tdata {data}")
    if cold:
        raise fail(f"cache is cold for: {', '.join(cold)}", EXIT_FAILURE)


DEFAULT_MODES = "vector,lexical,hybrid,hybrid+rerank"
DEFAULT_RESULTS = Path("out/eval")
QUICK_QUERIES = 50


def short_commit(cwd: Path | None = None) -> str:
    """`git rev-parse --short HEAD`, else `FOOTNOTE_BUILD_COMMIT`, else `unknown`."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607 - git from PATH
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        from footnote.bootstrap import build_info  # noqa: PLC0415 - heavy import

        return build_info().commit


def new_run_id(commit: str, now: datetime | None = None) -> str:
    """`<UTC timestamp>-<short commit>`, e.g. `20261020T084100Z-1a2b3c4`."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{commit}"


def _log(message: str) -> None:
    typer.echo(message, err=True)


def summary_lines(rows: list[Row]) -> list[str]:
    """One line per row: name, split, nDCG@10, R@10, p95."""
    lines = []
    for r in rows:
        mark = " *" if r.shipped else ""
        lines.append(
            f"{r.name}{mark}\t{r.split}\tnDCG@10 {r.metrics['ndcg_at_10']:.4f}"
            f"\tR@10 {r.metrics['recall_at_10']:.4f}\tp95 {r.p95_ms:.1f} ms"
        )
    return lines


async def _run_retrieval(  # noqa: PLR0913 - one value per command option
    settings: Settings,
    dataset_name: str,
    plan: str,
    split: str,
    *,
    modes: str,
    chunking: str | None,
    keep: bool,
    use_cache: bool,
    concurrency: int,
    limit: int | None,
    output_dir: Path,
) -> tuple[EvalOutcome, Path]:
    from footnote.app.evaluation.datasets import dataset_dir, load_beir  # noqa: PLC0415
    from footnote.app.evaluation.report import ReportMeta, write_reports  # noqa: PLC0415
    from footnote.app.evaluation.retrieval import (  # noqa: PLC0415
        EvalOptions,
        parse_chunkings,
        parse_modes,
        plan_ablation,
        plan_variants,
        run_retrieval_eval,
    )
    from footnote.app.evaluation.runtime import dataset_sha256, open_eval_runtime  # noqa: PLC0415

    dataset = load_beir(dataset_dir(dataset_name), dataset_name)
    commit = short_commit()
    run_id = new_run_id(commit)
    async with open_eval_runtime(settings, cache=use_cache) as rt:
        variants: list[Variant]
        if plan == "ablation":
            variants = plan_ablation(rt.shipped)
        else:
            chunkings = parse_chunkings(chunking) if chunking else [rt.shipped.chunking]
            variants = plan_variants(rt.shipped, parse_modes(modes), chunkings, split)
        warmed: list[str] = []

        def on_indexed(strategy: object, corpus: object) -> None:
            del corpus
            full = use_cache and strategy == rt.shipped.chunking
            if full and rt.mark_warm(dataset.name, len(dataset.corpus)):
                warmed.append(dataset.name)

        options = EvalOptions(
            variants=variants,
            shipped=rt.shipped,
            run_id=run_id,
            keep=keep,
            concurrency=concurrency,
            limit=limit,
            log=_log,
            on_indexed=on_indexed,
        )
        outcome = await run_retrieval_eval(rt.services, dataset, options)
        sha = dataset_sha256(dataset.name)
        lines = [
            f"dataset `{dataset.name}`"
            + (f" (archive sha256 {sha[:12]})" if sha else "")
            + f": {len(dataset.corpus)} documents",
            *rt.provenance,
            "FTS config `english`; binary relevance from qrels; depth 100, "
            "k_lexical = k_vector = max(configured, 100); documents pooled by first occurrence",
            "p50/p95: in-process `SearchService.search` latency per query (query embedding "
            "included, sequential queries)",
        ]
        for strategy, corpus in outcome.corpora.items():
            lines.append(
                f"{strategy} chunking: {corpus.chunks} chunks, {corpus.embedded} texts embedded "
                f"(cache misses and hits), indexed in {corpus.seconds:.0f}s"
            )
        if limit is not None:
            lines.append(f"first {limit} queries per split only")
        lines.append(
            f"embedding cache: {'on' if use_cache else 'off'}"
            + (f", marked warm: {', '.join(warmed)}" if warmed else "")
        )
        lines.append(
            f"temporary tenant {outcome.tenant} {'kept' if keep else 'deleted'}; "
            f"total {outcome.seconds:.0f}s"
        )
        directory = output_dir / run_id
        write_reports(directory, outcome.rows, ReportMeta(run_id, dataset.name, commit, lines))
    return outcome, directory


@eval_app.command("retrieval")
def retrieval(  # noqa: PLR0913, PLR0917 - one parameter per CLI option
    ctx: typer.Context,
    dataset: Annotated[str, typer.Option(help="BEIR dataset name (fetched).")] = "scifact",
    modes: Annotated[str, typer.Option(help="Comma-separated modes.")] = DEFAULT_MODES,
    chunking: Annotated[
        str | None, typer.Option(help="Comma-separated strategies (default: shipped).")
    ] = None,
    ablation: Annotated[
        bool, typer.Option(help="Run the ablation row set (eval/ablation.md).")
    ] = False,
    split: Annotated[str, typer.Option(help="qrels split: test or train.")] = "test",
    keep: Annotated[bool, typer.Option(help="Keep the temporary tenant.")] = False,
    cache: Annotated[bool, typer.Option(help="Use the embedding cache.")] = True,
    concurrency: Annotated[int, typer.Option(min=1, max=64, help="Documents in flight.")] = 8,
    limit: Annotated[int | None, typer.Option(min=1, help="First N queries per split.")] = None,
    quick: Annotated[
        bool, typer.Option(help=f"First {QUICK_QUERIES} queries per split (make eval-quick).")
    ] = False,
    output_dir: Annotated[Path, typer.Option(help="Results root.")] = DEFAULT_RESULTS,
) -> None:
    """Index a dataset into a temporary tenant, run every query per configuration and
    write `<output-dir>/<run_id>/retrieval.csv` and `ablation.md`."""
    from footnote.app.evaluation.bm25 import MissingExtraError, require_extra  # noqa: PLC0415
    from footnote.app.evaluation.retrieval import EvalError  # noqa: PLC0415
    from footnote.config import ConfigError  # noqa: PLC0415
    from footnote.domain.errors import DomainError  # noqa: PLC0415

    try:
        require_extra()
    except MissingExtraError as exc:
        raise fail(str(exc), EXIT_CODE_INVALID) from exc
    settings = settings_from(ctx)
    if quick and limit is None:
        limit = QUICK_QUERIES
    try:
        outcome, directory = asyncio.run(
            _run_retrieval(
                settings,
                dataset,
                "ablation" if ablation else "modes",
                split,
                modes=modes,
                chunking=chunking,
                keep=keep,
                use_cache=cache,
                concurrency=concurrency,
                limit=limit,
                output_dir=output_dir,
            )
        )
    except ConfigError as exc:
        raise fail(str(exc), EXIT_CODE_INVALID) from exc
    except (EvalError, FileNotFoundError, DomainError) as exc:
        raise fail(str(exc), EXIT_FAILURE) from exc
    for line in summary_lines(outcome.rows):
        typer.echo(line)
    typer.echo(f"wrote {directory / 'retrieval.csv'} and {directory / 'ablation.md'}")


# --- eval abstention / eval calibrate ---------------------------------------------------

RERANKER_FILES = ("all", "configured")


def _rerank_specs(
    settings: Settings, rt: EvalRuntime, which: str
) -> tuple[list[GateSpec], dict[str, float | None]]:
    """Rerank gates per reranker file (configured first) and each file's manifest value.

    `which` = `configured` scores the active file only; `all` adds every other variant file
    of the model present in `models.dir` (absent files are reported and skipped)."""
    from footnote.adapters.models import local_path, reranker_variant_key  # noqa: PLC0415
    from footnote.app.evaluation.abstention import GateSpec  # noqa: PLC0415
    from footnote.app.evaluation.retrieval import EvalError  # noqa: PLC0415
    from footnote.bootstrap import load_model_manifest, make_reranker  # noqa: PLC0415
    from footnote.domain.evidence import GateKind  # noqa: PLC0415

    if settings.rerank.provider != "onnx_cross_encoder" or rt.runtime.api.search.reranker is None:
        raise EvalError("the rerank gate needs rerank.provider onnx_cross_encoder")
    manifest = load_model_manifest(settings)
    entry = manifest.get(settings.rerank.model)
    active = reranker_variant_key(settings.rerank.variant)
    specs = [GateSpec(GateKind.RERANK, active, entry.variants[active].sha256)]
    current = {active: entry.variants[active].calibrated_threshold}
    if which == "all":
        for key, file in sorted(entry.variants.items()):
            if key == active:
                continue
            if not local_path(settings.models.dir, entry, file).is_file():
                _log(f"reranker file {key} ({file.path}) is not in models.dir; skipped")
                continue
            variant, _, arch = key.partition("-")
            reranker = make_reranker(settings, manifest, variant, arch or None)
            specs.append(GateSpec(GateKind.RERANK, key, file.sha256, reranker))
            current[key] = file.calibrated_threshold
    return specs, current


def _vector_spec(settings: Settings) -> GateSpec:
    from footnote.app.evaluation.abstention import GateSpec  # noqa: PLC0415
    from footnote.domain.evidence import GateKind  # noqa: PLC0415

    return GateSpec(GateKind.VECTOR, settings.embedding.default_space)


async def _run_abstention(  # noqa: PLR0913 - one value per command option
    settings: Settings,
    dataset_name: str,
    *,
    question_set: Path,
    keep: bool,
    use_cache: bool,
    concurrency: int,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    from footnote.app.evaluation import abstention as ab  # noqa: PLC0415
    from footnote.app.evaluation.datasets import dataset_dir, load_beir  # noqa: PLC0415
    from footnote.app.evaluation.runtime import open_eval_runtime  # noqa: PLC0415

    dataset = load_beir(dataset_dir(dataset_name), dataset_name)
    split = ab.abstention_split(dataset)
    try:
        ood = ab.load_question_set(question_set)
    except FileNotFoundError:
        ood = []
        _log(
            f"out-of-corpus set {question_set} not found; its curve is skipped "
            '(write 200 {{"id", "text"}} lines there to publish it)'
        )
    run_id = new_run_id(short_commit())
    directory = output_dir / run_id
    queries = dataset.queries
    groups = {
        ab.ANSWERABLE: [(q, queries[q]) for q in split.answerable],
        ab.UNANSWERABLE: [(q, queries[q]) for q in split.unanswerable],
    }
    async with open_eval_runtime(settings, cache=use_cache) as rt:
        rerank_specs, manifest_values = _rerank_specs(settings, rt, "configured")
        specs = [rerank_specs[0], _vector_spec(settings)]
        gate = rt.runtime.api.answers
        async with ab.gate_index(
            rt.services,
            dataset,
            split,
            run_id=run_id,
            chunking=rt.shipped.chunking,
            concurrency=concurrency,
            keep=keep,
            log=_log,
        ) as index:
            questions = [*groups[ab.ANSWERABLE], *groups[ab.UNANSWERABLE], *ood]
            scored = [await ab.score_gate(gate, index, questions, s, log=_log) for s in specs]
    rerank, vector = scored
    pos = {s.spec.kind.value: s.of(split.answerable) for s in scored}
    neg = {s.spec.kind.value: s.of(split.unanswerable) for s in scored}
    summaries = {
        name: ab.summarise(pos[name], neg[name], s.threshold or 0.0)
        for name, s in zip(("rerank", "vector"), scored, strict=True)
    }
    curves = {"": summaries["rerank"].curve, "vector_": summaries["vector"].curve}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "abstention_curve.csv").write_text(ab.curve_csv(curves), encoding="utf-8")
    labels = [s.spec.kind.value for s in scored]
    rows = [
        (group, qid, {s.spec.kind.value: s.scores[qid] for s in scored})
        for group, items in (*groups.items(), ("out_of_corpus", ood))
        for qid, _ in items
    ]
    (directory / "abstention_scores.csv").write_text(ab.scores_csv(rows, labels), encoding="utf-8")
    ood_summary: dict[str, Any] | None = None
    if ood:
        ood_ids = [q for q, _ in ood]
        ood_sums = {
            name: ab.summarise(pos[name], s.of(ood_ids), s.threshold or 0.0)
            for name, s in zip(("rerank", "vector"), scored, strict=True)
        }
        (directory / "abstention_curve_out_of_corpus.csv").write_text(
            ab.curve_csv({"": ood_sums["rerank"].curve, "vector_": ood_sums["vector"].curve}),
            encoding="utf-8",
        )
        ood_summary = {
            "path": str(question_set),
            "questions": len(ood),
            "rerank": ood_sums["rerank"].as_dict(),
            "vector": ood_sums["vector"].as_dict(),
        }
    head = summaries["rerank"].as_dict()
    summary: dict[str, Any] = {
        "run_id": run_id,
        "dataset": dataset.name,
        "gate": "rerank",
        "reranker_file": rerank.spec.key,
        "reranker_sha256": rerank.spec.sha256,
        "manifest_threshold": manifest_values[rerank.spec.key],
        **head,
        "per_group": {
            ab.ANSWERABLE: len(split.answerable),
            ab.UNANSWERABLE: len(split.unanswerable),
        },
        "skipped_claims": len(split.skipped),
        "index_documents": index.documents,
        "removed_documents": index.removed,
        "chunks": index.corpus.chunks,
        "vector": {"space": vector.spec.key, **summaries["vector"].as_dict()},
        "out_of_corpus": ood_summary,
        "files": ["abstention_curve.csv", "abstention_scores.csv"]
        + (["abstention_curve_out_of_corpus.csv"] if ood else []),
    }
    ab.write_json(directory / "abstention.json", summary)
    return summary, directory


@eval_app.command("abstention")
def abstention(  # noqa: PLR0913, PLR0917 - one parameter per CLI option
    ctx: typer.Context,
    dataset: Annotated[str, typer.Option(help="BEIR dataset name (fetched).")] = "scifact",
    unanswerable_set: Annotated[
        Path, typer.Option(help="Out-of-corpus questions (JSONL id/text).")
    ] = Path("eval/sets/unanswerable.jsonl"),
    keep: Annotated[bool, typer.Option(help="Keep the temporary tenant.")] = False,
    cache: Annotated[bool, typer.Option(help="Use the embedding cache.")] = True,
    concurrency: Annotated[int, typer.Option(min=1, max=64, help="Documents in flight.")] = 8,
    output_dir: Annotated[Path, typer.Option(help="Results root.")] = DEFAULT_RESULTS,
) -> None:
    """Gate scores of the two claim groups (no LLM call): write `abstention_curve.csv`,
    `abstention_scores.csv` and `abstention.json` (AUC, rates at the resolved threshold)."""
    from footnote.app.evaluation.retrieval import EvalError  # noqa: PLC0415
    from footnote.config import ConfigError  # noqa: PLC0415
    from footnote.domain.errors import DomainError  # noqa: PLC0415

    settings = settings_from(ctx)
    try:
        summary, directory = asyncio.run(
            _run_abstention(
                settings,
                dataset,
                question_set=unanswerable_set,
                keep=keep,
                use_cache=cache,
                concurrency=concurrency,
                output_dir=output_dir,
            )
        )
    except ConfigError as exc:
        raise fail(str(exc), EXIT_CODE_INVALID) from exc
    except (EvalError, FileNotFoundError, DomainError) as exc:
        raise fail(str(exc), EXIT_FAILURE) from exc
    for name, part in (("rerank", summary), ("vector", summary["vector"])):
        typer.echo(
            f"{name} gate: AUC {part['auc']:.4f}; at threshold {part['threshold']:.4f} "
            f"answer {part['answer_rate']:.2%} of answerable, "
            f"abstain {part['abstain_rate']:.2%} of unanswerable"
        )
    ood = summary["out_of_corpus"]
    if ood is not None:
        for name in ("rerank", "vector"):
            part = ood[name]
            typer.echo(
                f"out-of-corpus ({ood['questions']}) {name} gate: AUC {part['auc']:.4f}; "
                f"abstain {part['abstain_rate']:.2%} at {part['threshold']:.4f}"
            )
    typer.echo(f"wrote {directory / 'abstention_curve.csv'} and {directory / 'abstention.json'}")


async def _run_calibrate(  # noqa: PLR0913 - one value per command option
    settings: Settings,
    dataset_name: str,
    split_name: str,
    *,
    reranker_files: str,
    apply: bool,
    keep: bool,
    use_cache: bool,
    concurrency: int,
    output_dir: Path,
) -> tuple[dict[str, Any], Path, list[str]]:
    from footnote.app.evaluation import abstention as ab  # noqa: PLC0415
    from footnote.app.evaluation.datasets import dataset_dir, load_beir, query_ids  # noqa: PLC0415
    from footnote.app.evaluation.retrieval import EvalError  # noqa: PLC0415
    from footnote.app.evaluation.runtime import open_eval_runtime  # noqa: PLC0415

    dataset = load_beir(dataset_dir(dataset_name), dataset_name)
    if split_name not in dataset.qrels:
        raise EvalError(f"dataset {dataset.name} has no qrels split {split_name!r}")
    split = ab.abstention_split(dataset)
    indexed = frozenset(d.doc_id for d in ab.index_documents(dataset, split))
    gold = ab.gold_documents(dataset.qrels[split_name])
    ids = [q for q in query_ids(dataset, split_name) if q in dataset.queries]
    claims = ab.calibration_claims(ids, gold, indexed)
    if not claims:
        raise EvalError(f"no {split_name} claims have all gold documents in the index")
    target = settings.eval.calibration_target_answer_rate
    run_id = new_run_id(short_commit())
    directory = output_dir / run_id
    questions = [(q, dataset.queries[q]) for q in claims]
    applied: list[str] = []
    async with open_eval_runtime(settings, cache=use_cache) as rt:
        rerank_specs, manifest_values = _rerank_specs(settings, rt, reranker_files)
        vector_spec = _vector_spec(settings)
        spaces = {s.name: s for s in await rt.runtime.admin.all_spaces()}
        space_row = spaces.get(vector_spec.key)
        gate = rt.runtime.api.answers
        async with ab.gate_index(
            rt.services,
            dataset,
            split,
            run_id=run_id,
            chunking=rt.shipped.chunking,
            concurrency=concurrency,
            keep=keep,
            log=_log,
        ) as index:
            scored = [
                await ab.score_gate(gate, index, questions, s, log=_log)
                for s in [*rerank_specs, vector_spec]
            ]
        results = [
            ab.calibrate(
                s,
                claims,
                target,
                manifest_values.get(s.spec.key)
                if s.spec.kind.value == "rerank"
                else (space_row.calibrated_threshold if space_row else None),
            )
            for s in scored
        ]
        if apply:
            applied = await _apply_calibration(settings, rt, results)
    labels = [s.spec.label for s in scored]
    rows = [("calibration", q, {s.spec.label: s.scores[q] for s in scored}) for q in claims]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "calibration_scores.csv").write_text(ab.scores_csv(rows, labels), encoding="utf-8")
    summary: dict[str, Any] = {
        "run_id": run_id,
        "dataset": dataset.name,
        "split": split_name,
        "target_answer_rate": target,
        "claims": len(claims),
        "split_claims": len(ids),
        "index_documents": index.documents,
        "removed_documents": index.removed,
        "chunks": index.corpus.chunks,
        "rerank": [r.as_dict() for r in results if r.spec.kind.value == "rerank"],
        "vector": [r.as_dict() for r in results if r.spec.kind.value == "vector"],
        "applied": apply,
    }
    ab.write_json(directory / "calibration.json", summary)
    return summary, directory, applied


async def _apply_calibration(
    settings: Settings, rt: EvalRuntime, results: list[Calibration]
) -> list[str]:
    """Write the manifest values (reranker files, the space's model) and the space row."""
    from dataclasses import replace  # noqa: PLC0415

    from footnote.app.evaluation.manifest_edit import (  # noqa: PLC0415
        set_file_threshold,
        set_model_threshold,
    )
    from footnote.bootstrap import (  # noqa: PLC0415
        PROVIDER_ONNX,
        load_model_manifest,
        space_specs,
    )

    path = settings.models.manifest
    text = path.read_text(encoding="utf-8")
    lines: list[str] = []
    for r in results:
        if r.spec.kind.value == "rerank" and r.spec.sha256:
            text = set_file_threshold(text, r.spec.sha256, r.stored)
            lines.append(f"manifest {path}: reranker file {r.spec.key} = {r.stored:.4f}")
    space = next(s for s in settings.embedding.spaces if s.name == settings.embedding.default_space)
    vector = next(r for r in results if r.spec.kind.value == "vector")
    if space.provider == PROVIDER_ONNX:
        text = set_model_threshold(text, space.model, vector.stored)
        lines.append(f"manifest {path}: embedder {space.model} = {vector.stored:.4f}")
    path.write_text(text, encoding="utf-8")
    manifest = load_model_manifest(settings)  # the edited file must still load
    for r in results:
        if r.spec.kind.value == "rerank":
            file = manifest.get(settings.rerank.model).variants[r.spec.key]
            if file.calibrated_threshold != r.stored:
                msg = f"manifest edit of {r.spec.key} did not take effect"
                raise RuntimeError(msg)
    spec = next(s for s in space_specs(settings) if s.name == space.name)
    await rt.runtime.admin.register_space(replace(spec, calibrated_threshold=vector.stored))
    lines.append(f"embedding_spaces row {space.name}: calibrated_threshold = {vector.stored:.4f}")
    lines.append(
        f"config (overrides the space row, optional): embedding.spaces[name={space.name}]"
        f".calibrated_threshold: {vector.stored:.4f}"
    )
    active = next(r for r in results if r.spec.kind.value == "rerank")
    lines.append(
        f"config (optional; null uses the manifest value): "
        f"answer.evidence_threshold_rerank: {active.stored:.4f}"
    )
    return lines


@eval_app.command("calibrate")
def calibrate(  # noqa: PLR0913, PLR0917 - one parameter per CLI option
    ctx: typer.Context,
    dataset: Annotated[str, typer.Option(help="BEIR dataset name (fetched).")] = "scifact",
    split: Annotated[str, typer.Option(help="qrels split of the calibration claims.")] = "train",
    apply: Annotated[
        bool, typer.Option(help="Write the manifest values and the space's DB row.")
    ] = False,
    reranker_files: Annotated[
        str, typer.Option(help="Reranker files to calibrate: all (present) or configured.")
    ] = "all",
    keep: Annotated[bool, typer.Option(help="Keep the temporary tenant.")] = False,
    cache: Annotated[bool, typer.Option(help="Use the embedding cache.")] = True,
    concurrency: Annotated[int, typer.Option(min=1, max=64, help="Documents in flight.")] = 8,
    output_dir: Annotated[Path, typer.Option(help="Results root.")] = DEFAULT_RESULTS,
) -> None:
    """Calibrate the evidence-gate thresholds on the split's claims (abstention index) and
    write `calibration.json`; `--apply` updates the manifest and the space row."""
    from footnote.app.evaluation.manifest_edit import ManifestEditError  # noqa: PLC0415
    from footnote.app.evaluation.retrieval import EvalError  # noqa: PLC0415
    from footnote.config import ConfigError  # noqa: PLC0415
    from footnote.domain.errors import DomainError  # noqa: PLC0415

    if reranker_files not in RERANKER_FILES:
        raise fail(
            f"--reranker-files must be one of {', '.join(RERANKER_FILES)}", EXIT_CODE_INVALID
        )
    settings = settings_from(ctx)
    try:
        summary, directory, applied = asyncio.run(
            _run_calibrate(
                settings,
                dataset,
                split,
                reranker_files=reranker_files,
                apply=apply,
                keep=keep,
                use_cache=cache,
                concurrency=concurrency,
                output_dir=output_dir,
            )
        )
    except ConfigError as exc:
        raise fail(str(exc), EXIT_CODE_INVALID) from exc
    except (EvalError, FileNotFoundError, DomainError, ManifestEditError) as exc:
        raise fail(str(exc), EXIT_FAILURE) from exc
    typer.echo(
        f"{summary['claims']} of {summary['split_claims']} {split} claims; "
        f"target answer rate {summary['target_answer_rate']:.2f}"
    )
    for r in [*summary["rerank"], *summary["vector"]]:
        current = "none" if r["current"] is None else f"{r['current']:.4f}"
        typer.echo(
            f"{r['gate']} {r['key']}: threshold {r['stored']:.4f} (exact {r['threshold']:.6f}; "
            f"answers {r['answer_rate_at_stored']:.2%}); current {current}"
        )
    for line in applied:
        typer.echo(line)
    typer.echo(f"wrote {directory / 'calibration.json'}")


async def _run_answers(  # noqa: PLR0913 - one value per command option
    settings: Settings,
    dataset_name: str,
    *,
    limit: int,
    judge_model: str,
    keep: bool,
    use_cache: bool,
    concurrency: int,
    output_dir: Path,
) -> AnswerEvalResult:
    import httpx  # noqa: PLC0415

    from footnote.adapters.llm.client import LLMConfig, OpenAICompatLLM  # noqa: PLC0415
    from footnote.app.evaluation.answers import eval_answers  # noqa: PLC0415
    from footnote.app.evaluation.datasets import dataset_dir, load_beir  # noqa: PLC0415
    from footnote.app.evaluation.runtime import open_eval_runtime  # noqa: PLC0415

    dataset = load_beir(dataset_dir(dataset_name), dataset_name)
    run_id = new_run_id(short_commit())
    c = settings.llm
    judge_config = LLMConfig(
        base_url=c.base_url,
        model=judge_model,
        api_key=c.api_key.get_secret_value(),
        extra_headers=dict(c.extra_headers),
        extra_body=dict(c.extra_body),
        temperature=0.0,
        max_tokens=JUDGE_MAX_TOKENS,
        timeout=c.timeout,
        max_retries=c.max_retries,
    )
    async with (
        open_eval_runtime(settings, cache=use_cache) as rt,
        httpx.AsyncClient(timeout=c.timeout) as client,
    ):
        return await eval_answers(
            rt,
            dataset,
            limit=limit,
            judge=OpenAICompatLLM(judge_config, client),
            judge_model=judge_model,
            out_dir=output_dir / run_id,
            run_id=run_id,
            keep=keep,
            concurrency=concurrency,
            log=_log,
        )


JUDGE_MAX_TOKENS = 2048  # one JSON entry per sentence; 512 can cut it


@eval_app.command("answers")
def answers(  # noqa: PLR0913, PLR0917 - one parameter per CLI option
    ctx: typer.Context,
    judge_model: Annotated[str, typer.Option(help="Judge model id at llm.base_url.")],
    dataset: Annotated[str, typer.Option(help="BEIR dataset name (fetched).")] = "scifact",
    limit: Annotated[int, typer.Option(min=1, help="First N test query ids.")] = 100,
    keep: Annotated[bool, typer.Option(help="Keep the temporary tenant.")] = False,
    cache: Annotated[bool, typer.Option(help="Use the embedding cache.")] = True,
    concurrency: Annotated[int, typer.Option(min=1, max=64, help="Documents in flight.")] = 8,
    output_dir: Annotated[Path, typer.Option(help="Results root.")] = DEFAULT_RESULTS,
) -> None:
    """Generate answers for the first `limit` claims, then judge them: write
    `answers.csv` and `summary.md` (faithfulness, citation precision, answer rate)."""
    from footnote.app.evaluation.retrieval import EvalError  # noqa: PLC0415
    from footnote.config import ConfigError  # noqa: PLC0415
    from footnote.domain.errors import DomainError  # noqa: PLC0415

    settings = settings_from(ctx)
    if not settings.llm.base_url:
        raise fail("llm.base_url is empty (set FOOTNOTE_LLM_BASE_URL)", EXIT_CODE_INVALID)
    try:
        result = asyncio.run(
            _run_answers(
                settings,
                dataset,
                limit=limit,
                judge_model=judge_model,
                keep=keep,
                use_cache=cache,
                concurrency=concurrency,
                output_dir=output_dir,
            )
        )
    except ConfigError as exc:
        raise fail(str(exc), EXIT_CODE_INVALID) from exc
    except (EvalError, FileNotFoundError, DomainError) as exc:
        raise fail(str(exc), EXIT_FAILURE) from exc

    def ratio(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    typer.echo(
        f"faithfulness {ratio(result.faithfulness)}; citation precision "
        f"{ratio(result.citation_precision)}; answer rate {result.answer_rate:.2%} "
        f"(abstained {result.abstained}, unsupported {result.unsupported}); "
        f"judge parse failures {result.judge_parse_failures}"
    )
    if result.out_dir is not None:
        typer.echo(f"wrote {result.out_dir / 'summary.md'}")
