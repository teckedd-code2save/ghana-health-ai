"""Conservative source/content routing, not semantic or translation certification."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata


VERSION = "source-content-types-v2"
POLICY = {"version": VERSION, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
# These are vocabulary collections even though their names omit 'dictionary'.
LEXICAL = frozenset({
    "02_english-twi-dictionary", "03_science-glossary", "04_mathematics-glossary",
    "05_elementary_social_studies_twi", "06_hs_chemistry_twi",
    "07_hs_living_environment_twi_parallel", "08_hs_physics_twi_parallel",
    "09_Twi-Dictionary_parallel", "11_hs_us_history_twi_parallel",
    "12_hs_global_history_twi_parallel", "13_ela_twi_parallel",
    "16_Twi-census-Glossary_parallel",
    "17_A Dictionary of the Asante and Fante Language_mono",
    "20_Asante-Fante_Dictionary_12933_mono",
})
MIXED = frozenset({
    "01_medical_glossary", "10_ashantiproverbst00rattuoft_parallel",
    "14_Twi-Language-Guide_parallel", "15_twi-sample_parallel",
    "19_Gold Coast Akan Laws and Customs and the Akim Abuakwa Constitution_mono",
})
PROSE = frozenset({"youversion_bible_v1861", "youversion_bible_v1461"})
# Only route familiar complete short utterances; other short text needs review.
SHORT_UTTERANCES = frozenset({
    "hello", "hi", "good morning", "good afternoon", "good evening", "good night",
    "thank you", "thanks", "thank you very much", "you are welcome", "please help",
    "help me", "come here", "stop", "how are you", "and you", "i am fine",
    "i agree", "i disagree", "i don't know", "i do not know", "i understand",
    "i don't understand", "i do not understand", "yes", "no", "not yet",
})


def classify(english, twi, family, genre, declared_kind, origin):
    labels = sorted({part.strip() for part in genre.split(";") if part.strip()})
    words = re.findall(r"\w+", english)
    content_hash = hashlib.sha256(json.dumps([english, twi], ensure_ascii=False).encode()).hexdigest()
    result = {"policy": POLICY, "source_family": family, "source_labels": labels,
        "source_genre": genre, "declared_record_type": declared_kind, "origin": origin,
        "content_sha256": content_hash, "english_word_count": len(words),
        "effective_type": None, "view": None, "status": "review_needed",
        "reason": "unqualified_source_type", "semantic_certification": False}

    def routed(kind, view, reason):
        return {**result, "effective_type": kind, "view": view,
                "status": "automatically_screened", "reason": reason}

    if not english.strip() or not twi.strip():
        return {**result, "reason": "empty_reference"}
    if any(c == "\ufffd" or unicodedata.category(c) in ("Co", "Cs") or
           (unicodedata.category(c) == "Cc" and c not in "\n\r\t") for c in english + twi):
        return {**result, "reason": "nontext_codepoint"}
    if origin.startswith("synthetic"):
        return {**result, "reason": "synthetic_not_source_backed"}
    if family == "afriqa" and origin == "source_preserved" and declared_kind == "grounded_qa":
        return routed("grounded_qa", "questions", "source_grounded_question_not_response")
    if origin != "compiled_parallel_text":
        return result
    if family == "community_parallel":
        if labels and set(labels) <= LEXICAL:
            return routed("dictionary_entry", "lexicon", "audited_vocabulary_sources")
        if set(labels) & LEXICAL:
            return {**result, "reason": "conflicting_source_types"}
        if not labels or not set(labels) <= MIXED | PROSE:
            return {**result, "reason": "unknown_source_labels"}
        short = english.strip().casefold().replace("\u2019", "'").rstrip(".!?")
        if short in SHORT_UTTERANCES:
            return routed("utterance", "utterances", "recognized_short_utterance")
        if set(labels) & MIXED:
            return {**result, "reason": "mixed_source_requires_row_review"}
    elif family != "ghana_nlp_parallel" or labels != ["professional_parallel"]:
        return result
    short = english.strip().casefold().replace("\u2019", "'").rstrip(".!?")
    if short in SHORT_UTTERANCES:
        return routed("utterance", "utterances", "recognized_short_utterance")
    if len(words) < 4:
        return {**result, "reason": "short_reference_requires_row_review"}
    if re.search(r"\b(?:pl|inf|adj|adv)\.|(?:^|\s)-[nv]\.", english, re.I):
        return {**result, "reason": "lexical_notation_in_prose"}
    return routed("sentence", "sentences", "source_prose_with_content_screening")


def classify_source(source):
    return classify(source.get("english") or "", source["text"], source["source"],
                    source.get("genre", ""), source.get("original_record_type", source["record_type"]), source["origin"])


def validate_example_type(row, expected_view="sentences"):
    """Recompute the classification from actual supervision, never just a stamp."""
    assessment = row.get("type_assessment")
    if not assessment or assessment.get("policy") != POLICY:
        raise ValueError("Missing/stale record-type assessment; rebuild the handoff")
    try:
        source = json.loads(row["messages"][-2]["content"])["source_text"]
        target = row["messages"][-1]["content"]
        if row["target_language"] not in ("tw", "en"):
            raise ValueError("Unknown alignment target language")
        english, twi = (source, target) if row["target_language"] == "tw" else (target, source)
        expected = classify(english, twi, row["source_id"].split(":")[0],
            row["source_genre"], row["original_record_type"], row["origin"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Missing classification evidence") from error
    if assessment != expected or expected["view"] != expected_view or expected["effective_type"] != row["record_type"]:
        raise ValueError("Record-type/content mismatch; supervision cannot enter this view")
