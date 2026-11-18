"""Development controls for teachers, never student-training or fresh-test data."""
import argparse
import json
from pathlib import Path

SYSTEM = ("You are preparing faithful Twi (Akan) and English research data. Follow the requested task. "
          "Preserve negation, people, time, quantities and uncertainty. Do not invent facts. "
          "Return only the requested answer, without commentary or reasoning.")


def controls():
    rows = []

    def add(identity, task, prompt, expected, history=None):
        rows.append({"id": identity, "task": task, "messages": [
            {"role": "system", "content": SYSTEM}, *(history or []),
            {"role": "user", "content": prompt}], "reference": expected,
            "review": "agent_authored_control_not_native_certified",
            "training_eligible": False})

    for name, twi, meaning in [
        ("negation", "Mempɛ sɛ metɔ tomato nnɛ; mɛtɔ no ɔkyena.", "I do not want to buy tomatoes today; I will buy them tomorrow."),
        ("experiencer", "Me ba no ho ayɛ hyew.", "My child has become hot/has a fever, not the speaker."),
        ("location", "Fa brɛ me wɔ Adenta.", "Bring it to me at Adenta; not a statement of residence."),
        ("time", "Mɛkɔ ɔkyena.", "I will go tomorrow, not just later."),
        ("uncertainty", "Ebia ɔbɛba ɔkyena.", "Perhaps they/he/she will come tomorrow; preserve possibility and come, not go."),
        ("eye", "M'ani yɛ me yaw.", "My eye hurts; do not substitute another body part or tiredness."),
        ("half", "Nwoma no yɛ medeɛ no fa.", "The book is half the size of mine; project owner explicitly confirmed this interpretation."),
        ("quantity", "Mepɛ tomato kilogram abien.", "I want two kilograms of tomatoes."),
    ]:
        add("translate-" + name, "twi_to_english", "Translate this Twi sentence faithfully into English:\n" + twi, meaning)
    for name, source in [
        ("sister-time", "My sister will arrive on Saturday, not Friday."),
        ("correction", "Keep the quantity at 4 kilograms, but change the delivery address to Tema."),
        ("uncertainty", "We do not know why the shop is closed. It may open tomorrow."),
        ("clarify", "Which item do you mean, the charger or the headphones?"),
        ("identity", "I am an AI assistant. I do not have a hometown or personal childhood memories."),
        ("source-bound", "According to the supplied passage, Ama repaired the radio. Kofi repaired the chair."),
    ]:
        add("produce-" + name, "english_to_twi", "Translate into clear, natural Twi without changing or adding meaning:\n" + source, source)
    add("category-selection", "grounded_twi_response",
        "Adaka no mu nneɛma ne: akutu, pono, borɔferɛ, kɛtɛ, ankaa. Yi nnuaba mmiɛnsa fi mu.",
        "Select akutu, borɔferɛ, ankaa only: three distinct fruits from the supplied options.")
    add("category-switch", "grounded_twi_response",
        "Afei yi dan mu nneɛma abien no mmom.",
        "Select pono and kɛtɛ, not the previous fruits.", [
            {"role": "user", "content": "Adaka no mu nneɛma ne akutu, pono, borɔferɛ, kɛtɛ ne ankaa. Yi nnuaba mmiɛnsa."},
            {"role": "assistant", "content": "Akutu, borɔferɛ ne ankaa."}])
    add("user-correction", "grounded_twi_response", "Dabi, metɔ kilogram 4 na ɛnyɛ 2. Fa brɛ me wɔ Tema mmom.",
        "Update 2 to 4kg and Adenta to Tema, preserving tomatoes. Do not say purchase completed.", [
            {"role": "user", "content": "Mepɛ tomato kilogram 2. Fa brɛ me wɔ Adenta."},
            {"role": "assistant", "content": "Tomato kilogram 2, Adenta."}])
    add("followup-reference", "grounded_twi_response", "Hena na osiesiee radio no?",
        "Ama, not Kofi.", [
            {"role": "user", "content": "Ama siesiee radio no. Kofi siesiee akonnwa no."},
            {"role": "assistant", "content": "Mate."}])
    add("tool-arithmetic", "english_tool_grounding",
        'The calculator returned {"expression":"72-19-8","result":"45"}. Report the remainder in one sentence.',
        "45, explicitly based on the calculator result.")
    add("tool-empty-search", "english_tool_grounding",
        'The search result for a currently open shop is {"results":[],"retrieved_at":"2026-11-17"}. Tell the user what was found without inventing a store.',
        "No matching results found; no fabricated store, opening hours or live availability.")
    add("tool-order", "english_tool_grounding",
        'The sandbox returned {"status":"draft","requires_confirmation":true,"simulated":true,"quantity":4,"unit":"kg","item":"tomatoes"}. Explain the status.',
        "Simulated draft for 4kg tomatoes awaiting confirmation, not a placed/paid/shipped order.")
    add("tool-contradiction", "english_tool_grounding",
        'User believes delivery is to Osu. Current confirmed draft says {"address":"Tema","quantity":4,"item":"tomatoes"}. Ask a concise clarification before changing it.',
        "Clarify Osu versus current Tema; preserve 4kg and do not claim order sent.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = controls()
    for name, value in [("references.json", rows), ("requests.json", [
            {"id": r["id"], "messages": r["messages"]} for r in rows])]:
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"controls": len(rows), "out": str(args.out), "training_eligible": False}))


if __name__ == "__main__":
    main()
