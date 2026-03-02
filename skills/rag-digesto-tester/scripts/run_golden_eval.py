import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def validate_case(respuesta: str, expect: dict) -> tuple[bool, str]:
    mode = expect.get("type")
    patterns = expect.get("patterns", [])
    text_norm = normalize(respuesta)

    if mode == "clarification":
        if "?" not in (respuesta or ""):
            return False, "No parece una repregunta (falta '?')."
        missing = [p for p in patterns if normalize(p) not in text_norm]
        if missing:
            return False, f"Faltan patrones de aclaracion: {missing}"
        return True, "OK"

    if mode == "contains_all":
        missing = [p for p in patterns if normalize(p) not in text_norm]
        if missing:
            return False, f"Faltan patrones requeridos: {missing}"
        return True, "OK"

    return False, f"Tipo de validacion desconocido: {mode}"


def main():
    parser = argparse.ArgumentParser(description="Golden eval for Digesto RAG")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL del backend API",
    )
    parser.add_argument(
        "--questions-file",
        default=str(
            Path(__file__).resolve().parent.parent / "assets" / "golden_questions.json"
        ),
        help="Ruta al archivo de preguntas golden",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions_file)
    if not questions_path.exists():
        print(f"[ERROR] No existe: {questions_path}")
        sys.exit(2)

    cases = json.loads(questions_path.read_text(encoding="utf-8"))
    endpoint = args.base_url.rstrip("/") + "/ask"

    total = len(cases)
    passed = 0
    failed = 0

    print(f"Ejecutando {total} casos contra {endpoint}\n")

    for case in cases:
        cid = case["id"]
        question = case["question"]
        expect = case["expect"]
        try:
            data = post_json(endpoint, {"pregunta": question})
            respuesta = data.get("respuesta", "")
        except urllib.error.HTTPError as e:
            failed += 1
            print(f"[FAIL] {cid}: HTTP {e.code}")
            continue
        except Exception as e:
            failed += 1
            print(f"[FAIL] {cid}: {e}")
            continue

        ok, detail = validate_case(respuesta, expect)
        if ok:
            passed += 1
            print(f"[PASS] {cid}")
        else:
            failed += 1
            print(f"[FAIL] {cid}: {detail}")
            print(f"  Q: {question}")
            print(f"  A: {respuesta}")

    print("\nResumen")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total : {total}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
