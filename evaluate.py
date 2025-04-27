import argparse
import os
import json
from datetime import datetime
import random

def dummy_evaluate(model_path, model_type, model_format):
    if model_type == "STT":
        return {
            "WER": round(random.uniform(0.05, 0.25), 3),
            "CER": round(random.uniform(0.03, 0.15), 3),
            "Latency(ms)": random.randint(50, 200),
            "Throughput(req/s)": round(random.uniform(5, 15), 2)
        }
    elif model_type == "NLU":
        return {
            "Accuracy": round(random.uniform(0.85, 0.95), 3),
            "F1": round(random.uniform(0.85, 0.95), 3),
            "Latency(ms)": random.randint(50, 200),
            "Throughput(req/s)": round(random.uniform(5, 15), 2)
        }
    else:
        return {
            "EER": round(random.uniform(0.02, 0.06), 3),
            "Latency(ms)": random.randint(50, 150),
            "Throughput(req/s)": round(random.uniform(10, 20), 2)
        }

def save_result(original_name, model_type, model_format, metrics):
    result = {
        "ModelName": original_name,
        "ModelFormat": model_format,
        "Dataset": "SampleDataset",
        **metrics,
        "Timestamp": datetime.now().isoformat()
    }

    save_dir = {
        "STT": "results/stt",
        "NLU": "results/nlu",
        "인증": "results/auth"
    }[model_type]

    json_filename = original_name.replace(".", "_") + "_result.json"
    save_path = os.path.join(save_dir, json_filename)

    os.makedirs(save_dir, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"✅ 평가 결과 저장 완료: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="평가할 모델 파일 경로 (temp 파일)")
    parser.add_argument("--model_type", type=str, required=True, help="모델 종류 (STT, NLU, 인증)")
    parser.add_argument("--model_format", type=str, required=True, help="모델 포맷 (onnx, gguf, cpp)")
    parser.add_argument("--original_name", type=str, required=True, help="업로드한 원본 파일명")
    args = parser.parse_args()

    metrics = dummy_evaluate(args.model_path, args.model_type, args.model_format)
    save_result(args.original_name, args.model_type, args.model_format, metrics)
