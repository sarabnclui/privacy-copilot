import csv, os, time
from privacy_copilot.rag.answer import answer

def has_expected_source(resp: str, expected: str) -> bool:
    expected = expected.lower()
    return expected in resp.lower()

def main():
    path = "eval/qa_gold.csv"
    if not os.path.exists(path):
        print("No eval/qa_gold.csv found.")
        return
    total = passed = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row["question"].strip().strip('"')
            expected = row["expected_source"].strip().strip('"')
            print(f"\nQ: {q}")
            try:
                resp = answer(q)
            except Exception as e:
                print(f"Error answering: {e}")
                continue
            ok = has_expected_source(resp, expected)
            print("PASS" if ok else "FAIL")
            if not ok:
                print("— expected to see:", expected)
                print("— got:\n", resp)
            total += 1
            passed += int(ok)
            time.sleep(0.5)  # polite pacing
    print(f"\nSummary: {passed}/{total} passed")

if __name__ == "__main__":
    # Skip if no key set
    if not (os.getenv("OPENAI_API_KEY")):
        print("OPENAI_API_KEY not set; skipping eval.")
    else:
        main()
