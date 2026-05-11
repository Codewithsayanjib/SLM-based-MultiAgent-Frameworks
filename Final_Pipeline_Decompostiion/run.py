from slm_comm.config import load_config
from slm_comm.data import load_dataset_records
from slm_comm.model import HuggingFaceTextGenerator
from slm_comm.pipeline import build_pipeline
from slm_comm.metrics import summarize_results
from slm_comm.utils import ensure_dir, save_json, set_seed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run compact multi-agent SLM experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    model = HuggingFaceTextGenerator(cfg.model, cfg.generation)
    dataset = load_dataset_records(cfg.dataset, seed=cfg.seed)
    pipeline = build_pipeline(cfg, model)

    rows = []
    for idx, example in enumerate(dataset):
        try:
            result = pipeline.run(example)
        except Exception as exc:
            print(f"[{idx + 1}/{len(dataset)}] ERROR: {exc}")
            result = {
                "planner_text": None,
                "communication": None,
                "solver_text": None,
                "verifier_text": None,
                "prediction": "",
                "correct": False,
                "token_cost": 0,
                "planner_tokens": 0,
                "solver_tokens": 0,
                "verifier_tokens": 0,
                "error": str(exc),
            }
        row = {
            "index": idx,
            "question": example["question"],
            "gold_answer": example["answer"],
            **result,
        }
        rows.append(row)
        print(f"[{idx + 1}/{len(dataset)}] correct={row['correct']} tokens={row['token_cost']}")
        if (idx + 1) % 10 == 0 or (idx + 1) == len(dataset):
            save_json(f"{cfg.output_dir}/predictions.json", rows)

    summary = summarize_results(rows)
    save_json(f"{cfg.output_dir}/summary.json", summary)
    print("\nSummary")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
