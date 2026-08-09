"""Parser tests against small synthetic raw trees under the tmp data root
(the conftest fixture redirects GLYPHOS_DATA_ROOT per test)."""

import pytest

from glyphos.data.ingest import NotIngestable, cognates, coptic, hebrew, logogram, stubs
from glyphos.data.registry import CORPORA, ready_corpora
from glyphos.data.schema import Record, read_records, write_records
from glyphos.utils import paths

# -- schema -----------------------------------------------------------------


def test_record_roundtrip(tmp_path):
    recs = [
        Record("c", "d1", "s1", {"text": "ⲡⲃⲓⲟⲥ 𓐩𓏌𓀜"}, {"k": 1}),
        Record("c", "d1", "s2", {"text": "two"}, {}),
    ]
    path = tmp_path / "r.jsonl"
    assert write_records(recs, path) == 2
    assert list(read_records(path)) == recs


# -- coptic -----------------------------------------------------------------


COPTIC_CONLLU = """# newdoc id = mycorpus:doc.01
# sent_id = doc01_s1
# text_en = The life.
# text = ⲡⲃⲓⲟⲥ ⲁⲩⲱ
1\tⲡ\tⲡ\tDET\tART\t_\t2\tdet\t_\t_
2\tⲃⲓⲟⲥ\tⲃⲓⲟⲥ\tNOUN\tN\t_\t0\troot\t_\t_

# sent_id = doc01_s2
# text = ⲧⲡⲟⲗⲓⲧⲉⲓⲁ
1\tⲧⲡⲟⲗⲓⲧⲉⲓⲁ\t_\t_\t_\t_\t_\t_\t_\t_
"""


def test_coptic_parser(tmp_path):
    target = paths.data_root() / "coptic_scriptorium/raw/corpora/AP/ap_CONLLU"
    target.mkdir(parents=True)
    (target / "a.conllu").write_text(COPTIC_CONLLU, encoding="utf-8")
    recs = list(coptic.parse())
    assert len(recs) == 2
    assert recs[0].doc_id == "mycorpus:doc.01"
    assert recs[0].fields == {"text": "ⲡⲃⲓⲟⲥ ⲁⲩⲱ", "translation_en": "The life."}
    assert recs[0].meta == {"subcorpus": "AP", "dialect": "sahidic"}
    # second sentence has no blank line after it (EOF flush) and no text_en
    assert recs[1].sent_id == "doc01_s2"
    assert "translation_en" not in recs[1].fields


def test_coptic_missing_raw_is_loud():
    with pytest.raises(NotIngestable, match="git clone"):
        list(coptic.parse())


# -- hebrew -----------------------------------------------------------------


HEBREW_OSIS = """<?xml version="1.0" encoding="UTF-8"?>
<osis xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
 <osisText xml:lang="he">
  <div type="book" osisID="Gen">
   <chapter osisID="Gen.1">
    <verse osisID="Gen.1.1"><w lemma="b">בְּ</w><w lemma="r">רֵאשִׁית</w></verse>
    <verse osisID="Gen.1.2"><w lemma="v">וְהָאָרֶץ</w></verse>
    <verse osisID="Gen.1.3"><seg>empty verse no words</seg></verse>
   </chapter>
  </div>
 </osisText>
</osis>
"""


def test_hebrew_parser(tmp_path):
    wlc = paths.data_root() / "hebrew_morphhb/raw/morphhb/wlc"
    wlc.mkdir(parents=True)
    (wlc / "Gen.xml").write_text(HEBREW_OSIS, encoding="utf-8")
    recs = list(hebrew.parse())
    assert [r.sent_id for r in recs] == ["Gen.1.1", "Gen.1.2"]  # wordless verse skipped
    assert recs[0].doc_id == "Gen"
    assert recs[0].fields["text"] == "בְּ רֵאשִׁית"


# -- cognates ---------------------------------------------------------------


def test_cognates_parser(tmp_path):
    data = paths.data_root() / "ugaritic_hebrew_cognates/raw/NeuroDecipher/data"
    data.mkdir(parents=True)
    (data / "uga-heb.no_spe.cog").write_text(
        "uga-no_spe\theb-no_spe\nab\tab\nib\tawyb\n", encoding="utf-8"
    )
    recs = list(cognates.parse_for("ugaritic_hebrew_cognates"))
    assert len(recs) == 2
    assert recs[0].fields == {"src": "ab", "tgt": "ab"}
    assert recs[0].meta == {"src_name": "uga", "tgt_name": "heb"}


def test_cognates_bad_format_is_loud(tmp_path):
    data = paths.data_root() / "ugaritic_hebrew_cognates/raw/NeuroDecipher/data"
    data.mkdir(parents=True)
    (data / "uga-heb.no_spe.cog").write_text("only-one-column\nx\n", encoding="utf-8")
    with pytest.raises(ValueError, match="2-column"):
        list(cognates.parse_for("ugaritic_hebrew_cognates"))


# -- logogram inventory -----------------------------------------------------


def test_logogram_inventory(tmp_path):
    data = paths.data_root() / "logogram_nlp/raw/logogramNLP/data"
    (data / "LNA").mkdir(parents=True)
    (data / "LNA" / "img1.png").write_bytes(b"x" * 10)
    (data / "LNA" / ".DS_Store").write_bytes(b"junk")
    (data / "EGY").mkdir()
    (data / "EGY" / "meta.json").write_text("{}")
    inv = logogram.inventory()
    assert inv["LNA"] == {"files": 1, "bytes": 10, "by_extension": {".png": 1}}
    assert inv["EGY"]["files"] == 1


# -- registry / stubs -------------------------------------------------------


def test_stubs_raise_with_unblock_path():
    for name in ("linear_b_damos", "mayan", "libyco_berber"):
        with pytest.raises(NotIngestable):
            stubs.parse_for(name)


def test_registry_is_complete_and_consistent():
    assert set(ready_corpora()) == {
        "tla_earlier_egyptian",
        "tla_late_egyptian",
        "tla_demotic",
        "coptic_scriptorium",
        "hebrew_morphhb",
        "ugaritic_hebrew_cognates",
        "linearb_greek_cognates",
        "logogram_nlp",
        "greek_first1k",
        "cuneiform_cdli",
        "meroitic_rem",
        "logogram_egy",
    }
    for name, spec in CORPORA.items():
        meta = spec.meta()
        assert meta.name == name
        for scheme in spec.schemes:
            assert scheme in {
                "random",
                "document_heldout",
                "dedup",
                "period_heldout",
                "sign_heldout",
            }
        if "sign_heldout" in spec.schemes:
            assert meta.primary_field == "hieroglyphs"


def test_census_renders_pending_and_ingested(tmp_path):
    import json

    from glyphos.data.census import build_census, render_markdown

    processed = paths.data_root() / "coptic_scriptorium/processed"
    processed.mkdir(parents=True)
    (processed / "manifest.json").write_text(
        json.dumps(
            {"n_records": 5, "n_docs": 2, "n_tokens_primary": 40, "data_version": "abc123def456"}
        )
    )
    census = build_census()
    by_name = {r["corpus"]: r for r in census["rows"]}
    assert by_name["coptic_scriptorium"]["status"] == "ingested"
    assert by_name["tla_earlier_egyptian"]["status"] == "pending"
    assert by_name["linear_b_damos"]["status"] == "stub"
    text = render_markdown(census)
    assert "abc123def456" in text and "linear_b_damos" in text


# -- new-corpus parsers (greek / cdli / meroitic) ---------------------------


GREEK_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
 <teiHeader><fileDesc><titleStmt><title>t</title></titleStmt></fileDesc></teiHeader>
 <text><body><div type="edition">
  <p>ἐν ἀρχῇ ἦν ὁ λόγος</p>
  <p>too short</p>
  <l>μῆνιν ἄειδε θεὰ Πηληϊάδεω</l>
 </div></body></text>
</TEI>
"""


def test_greek_parser(tmp_path):
    from glyphos.data.ingest import greek

    work = paths.data_root() / "greek_first1k/raw/First1KGreek/data/tlg0012/tlg001"
    work.mkdir(parents=True)
    (work / "tlg0012.tlg001.1st1K-grc1.xml").write_text(GREEK_TEI, encoding="utf-8")
    (work / "tlg0012.tlg001.1st1K-eng1.xml").write_text(GREEK_TEI, encoding="utf-8")
    recs = list(greek.parse())
    # eng file skipped by filename; short/non-Greek <p> skipped by content
    assert len(recs) == 2
    assert recs[0].doc_id == "tlg0012.tlg001"
    assert "λόγος" in recs[0].fields["text"]


CDLI_ATF = """&P100001 = Test tablet A
#atf: lang akk
@tablet
@obverse
1. um-ma {d}utu-szi
2. a-na be-li2-ia
$ rest broken
&P100002 = Test tablet B
@tablet
1. 1(disz) sila3 kasz
"""


def test_cdli_parser(tmp_path):
    from glyphos.data.ingest import cdli

    raw = paths.data_root() / "cuneiform_cdli/raw"
    raw.mkdir(parents=True)
    (raw / "cdliatf_unblocked.atf").write_text(CDLI_ATF, encoding="utf-8")
    (raw / "cdli_cat.csv").write_text(
        "id_text,language,period\nP100001,Akkadian,Old Babylonian\nP100002,Sumerian,Ur III\n",
        encoding="utf-8",
    )
    recs = list(cdli.parse())
    assert [r.doc_id for r in recs] == ["P100001", "P100002"]
    assert recs[0].fields["text"] == "um-ma {d}utu-szi / a-na be-li2-ia"
    assert recs[0].meta["language"] == "Akkadian"
    assert recs[1].meta["period"] == "Ur III"


def test_meroitic_parser(tmp_path):
    from glyphos.data.ingest import meroitic

    data = paths.data_root() / "meroitic_rem/raw/Meroitic-Corpus/Data"
    data.mkdir(parents=True)
    for name in meroitic.DOC_FILES.values():
        (data / name).write_text("kiSri menEtel\n\nwES wetNyi\n", encoding="utf-8")
    recs = list(meroitic.parse())
    assert len(recs) == 2 * len(meroitic.DOC_FILES)
    assert {r.doc_id for r in recs} == set(meroitic.DOC_FILES)
    assert recs[0].fields["text"] == "kiSri menEtel"
