#!/usr/bin/env python
"""Data acquisition, census, splits, and freezing (Phase 1).

    prepare_data.py ingest (--corpus NAME ... | --all)
    prepare_data.py census
    prepare_data.py split --corpus NAME (--scheme S | --all-schemes)
                          [--tag v1] [--seed 1234]
    prepare_data.py freeze --corpus NAME --scheme S [--tag v1]
    prepare_data.py verify

Raw downloads (git clones / HF datasets) are documented per corpus in
glyphos/data/ingest/*; a missing raw tree fails loudly with the exact fetch
command. Split test partitions become immutable via `freeze` — enforced by
the open() guard from then on.
"""

import argparse
import dataclasses
import json
import sys
from datetime import UTC, datetime

from glyphos.data import freeze
from glyphos.data.guard import install_guard
from glyphos.data.ingest import NotIngestable, processed_dir
from glyphos.data.registry import CORPORA, CorpusSpec
from glyphos.data.schema import read_records, write_records
from glyphos.data.splits import (
    DEFAULT_RATIOS,
    dedup_against_train,
    split_document,
    split_period,
    split_random,
    split_sign_heldout,
)
from glyphos.utils.hashing import hash_file


def _spec(corpus: str) -> CorpusSpec:
    if corpus not in CORPORA:
        raise SystemExit(f"unknown corpus {corpus!r}; known: {', '.join(sorted(CORPORA))}")
    return CORPORA[corpus]


# -- ingest -----------------------------------------------------------------


def _uniquify_sent_ids(records):
    seen = set()
    for rec in records:
        sid = rec.sent_id
        if sid in seen:
            k = 2
            while f"{sid}#{k}" in seen:
                k += 1
            rec = dataclasses.replace(rec, sent_id=f"{sid}#{k}")
        seen.add(rec.sent_id)
        yield rec


def _ingest_records(spec: CorpusSpec, meta, records_path) -> tuple[int, int, int, dict]:
    docs: set[str] = set()
    field_cov: dict[str, int] = {}
    stats = {"tokens": 0}

    def stream():
        for rec in _uniquify_sent_ids(spec.parse()):
            docs.add(rec.doc_id)
            if meta.primary_field and meta.primary_field in rec.fields:
                stats["tokens"] += len(rec.fields[meta.primary_field].split())
            for name in rec.fields:
                field_cov[name] = field_cov.get(name, 0) + 1
            yield rec

    n = write_records(stream(), records_path)
    return n, len(docs), stats["tokens"], dict(sorted(field_cov.items()))


def cmd_ingest(args: argparse.Namespace) -> int:
    targets = sorted(CORPORA) if args.all else args.corpus
    if not targets:
        raise SystemExit("ingest: pass --corpus NAME (repeatable) or --all")
    failures = []
    for corpus in targets:
        spec = _spec(corpus)
        meta = spec.meta()
        if meta.kind == "stub":
            print(f"[ingest] SKIP {corpus}: stub — {meta.notes}")
            continue
        out = processed_dir(corpus)
        out.mkdir(parents=True, exist_ok=True)
        manifest: dict = {
            "corpus": corpus,
            "kind": meta.kind,
            "source": meta.source,
            "license": meta.license,
            "encoding": meta.encoding,
            "primary_field": meta.primary_field,
            "translation_field": meta.translation_field,
            "translation_lang": meta.translation_lang,
            "notes": meta.notes,
            "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        try:
            if spec.parse is None:
                manifest["inventory"] = spec.inventory()
                manifest["data_version"] = "inventory"
                print(f"[ingest] {corpus}: inventory of {len(manifest['inventory'])} subsets")
            else:
                records_path = out / "records.jsonl"
                n, n_docs, n_tokens, field_cov = _ingest_records(spec, meta, records_path)
                if n == 0:
                    raise NotIngestable(f"{corpus}: parser produced 0 records")
                manifest.update(
                    {
                        "n_records": n,
                        "n_docs": n_docs,
                        "n_tokens_primary": n_tokens,
                        "field_coverage": field_cov,
                        "data_version": hash_file(records_path),
                    }
                )
                print(
                    f"[ingest] {corpus}: {n:,} records, {n_docs:,} docs, "
                    f"{n_tokens:,} primary tokens, data_version={manifest['data_version']}"
                )
        except NotIngestable as exc:
            print(f"[ingest] FAILED {corpus}: {exc}", file=sys.stderr)
            failures.append(corpus)
            continue
        with open(out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
    if failures:
        print(f"[ingest] failures: {failures}", file=sys.stderr)
        return 1
    return 0


# -- census -----------------------------------------------------------------


def cmd_census(_args: argparse.Namespace) -> int:
    from glyphos.data.census import write_census

    out = write_census()
    print(f"[census] wrote {out} (+ census.json)")
    return 0


# -- split ------------------------------------------------------------------


def _apply_scheme(scheme: str, records: list, meta, seed: int):
    if scheme == "random":
        return split_random(records, DEFAULT_RATIOS, seed), {}
    if scheme == "document_heldout":
        return split_document(records, DEFAULT_RATIOS, seed), {}
    if scheme == "dedup":
        base = split_document(records, DEFAULT_RATIOS, seed)
        # target side for parallel corpora (full coverage); primary text for
        # monolingual ones (translations there are partial, e.g. Coptic text_en)
        target = meta.translation_field if meta.kind == "parallel" else meta.primary_field
        deduped, report = dedup_against_train(base, target_field=target)
        return deduped, {
            "dedup_target_field": target,
            "dedup_checked": report.checked,
            "dedup_removed_exact": report.removed_exact,
            "dedup_removed_near": report.removed_near,
        }
    if scheme == "period_heldout":
        return split_period(records, DEFAULT_RATIOS, seed), {}
    if scheme == "sign_heldout":
        result = split_sign_heldout(records, meta.primary_field, DEFAULT_RATIOS, seed)
        return result.splits, {"held_out_signs": result.held_out_signs}
    raise SystemExit(f"unknown scheme {scheme!r}")


def cmd_split(args: argparse.Namespace) -> int:
    spec = _spec(args.corpus)
    meta = spec.meta()
    schemes = list(spec.schemes) if args.all_schemes else [args.scheme]
    if not schemes or schemes == [None]:
        raise SystemExit(f"split: pass --scheme (one of {spec.schemes}) or --all-schemes")
    records_path = processed_dir(args.corpus) / "records.jsonl"
    if not records_path.exists():
        raise SystemExit(f"{args.corpus} not ingested yet: run prepare_data.py ingest first")
    records = list(read_records(records_path))
    for scheme in schemes:
        if scheme not in spec.schemes:
            raise SystemExit(f"scheme {scheme!r} not applicable to {args.corpus}: {spec.schemes}")
        splits, extra = _apply_scheme(scheme, records, meta, args.seed)
        version, out_dir = freeze.write_split(splits, args.corpus, scheme, args.tag)
        info = {
            "corpus": args.corpus,
            "scheme": scheme,
            "tag": args.tag,
            "seed": args.seed,
            "split_version": version,
            "sizes": {p: len(splits[p]) for p in ("train", "valid", "test")},
            **extra,
        }
        with open(out_dir / "split_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        print(f"[split] {args.corpus}/{scheme}/{args.tag}: {info['sizes']} version={version}")
        for key in ("dedup_removed_exact", "dedup_removed_near"):
            if key in extra:
                print(f"[split]   {key}={extra[key]}")
    return 0


# -- freeze / verify --------------------------------------------------------


def cmd_freeze(args: argparse.Namespace) -> int:
    digest = freeze.freeze_split(args.corpus, args.scheme, args.tag)
    print(f"[freeze] {args.corpus}/{args.scheme}/{args.tag}/test frozen: sha256={digest[:16]}…")
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    failures = freeze.verify_all()
    if failures:
        for failure in failures:
            print(f"[verify] VIOLATION: {failure}", file=sys.stderr)
        return 1
    print("[verify] all frozen test partitions intact")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="raw -> normalized records + manifest")
    p_ing.add_argument("--corpus", action="append", default=[])
    p_ing.add_argument("--all", action="store_true")

    sub.add_parser("census", help="write docs/census/census.{md,json}")

    p_spl = sub.add_parser("split", help="materialize a split scheme")
    p_spl.add_argument("--corpus", required=True)
    p_spl.add_argument("--scheme", default=None)
    p_spl.add_argument("--all-schemes", action="store_true")
    p_spl.add_argument("--tag", default="v1")
    p_spl.add_argument("--seed", type=int, default=1234)

    p_frz = sub.add_parser("freeze", help="freeze a split's test partition")
    p_frz.add_argument("--corpus", required=True)
    p_frz.add_argument("--scheme", required=True)
    p_frz.add_argument("--tag", default="v1")

    sub.add_parser("verify", help="re-hash all frozen test partitions")

    args = parser.parse_args(argv)
    install_guard()
    commands = {
        "ingest": cmd_ingest,
        "census": cmd_census,
        "split": cmd_split,
        "freeze": cmd_freeze,
        "verify": cmd_verify,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
