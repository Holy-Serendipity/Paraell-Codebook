import json
import math
import sys
import os


def parse_recommendations(input_path: str, output_path: str = None):
    """Parse recommendations JSON into {user_id: {item_id: score, ...}} format.

    Args:
        input_path: Path to recommendations JSON file.
        output_path: Optional output file path. If not provided, uses input name
                     with '_parsed' suffix.
    """
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    print(f"Loading recommendations from: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    recommendations = data.get('recommendations', [])
    total = len(recommendations)

    if total == 0:
        print("ERROR: No recommendations found in file.")
        sys.exit(1)

    print(f"Total users: {total}")

    # Build {user_id: {item_id: score, ...}} format
    result = {}
    for entry in recommendations:
        user_id = entry['user_id']
        recs = entry['recommendations']
        result[user_id] = {rec['item_id']: math.exp(rec['score']) for rec in recs}
    # Check top_k consistency
    sample_n = len(next(iter(result.values())))
    print(f"Top-K per user: {sample_n}")

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_parsed{ext}"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved parsed result to: {output_path}")
    print(f"Format: {{user_id: {{item_id: score, ...}}, ...}}  ({len(result)} users)")

    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python parse_recommendations.py <input.json> [output.json]")
        print("Example: python parse_recommendations.py recommendations.json")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    parse_recommendations(input_path, output_path)
