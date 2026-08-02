import os
import json


PROMPT_PATH = ("src/reporting/prompts/report_prompt.txt")
DEFAULT_LLM_MODEL = "gpt-5"


def load_prompt_template():

    with open(PROMPT_PATH, "r") as file:
        return file.read()


def build_template_report(findings):

    report = f"""
# MRI Analysis Report

## 1. Prediction Summary

The model predicts:
**{findings['predicted_class']}**

---

## 2. Confidence Assessment

- Confidence Score:
  {findings['confidence_score']}

- Confidence Level:
  {findings['confidence_level']}

---

## 3. Interpretability Notes

{findings['interpretation_summary']}

Attention Quality:
**{findings['attention_quality']}**

Grad-CAM Visualization:
{findings['gradcam_path']}

---

## 4. Limitations

- Research-use only
- Not intended for clinical diagnosis
- Requires specialist review

---

## 5. Recommendation

{findings['review_recommendation']}
"""

    return report.strip()


def build_llm_report(findings, model=None):

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required for LLM report generation. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Falling back to the template report."
        )

    prompt_template = load_prompt_template()

    findings_json = json.dumps(
        {
            "predicted_class": findings["predicted_class"],
            "confidence_score": findings["confidence_score"],
            "confidence_level": findings["confidence_level"],
            "probabilities": findings["probabilities"],
            "attention_quality": findings["attention_quality"],
            "gradcam_path": findings["gradcam_path"],
            "interpretation_summary": findings["interpretation_summary"],
            "review_recommendation": findings["review_recommendation"],
            "limitations": findings["limitations"],
        },
        indent=2
    )

    formatted_prompt = prompt_template.format(
        findings_json=findings_json
    )

    client = OpenAI()

    response = client.responses.create(
        model=model or os.getenv("OPENAI_REPORT_MODEL", DEFAULT_LLM_MODEL),
        instructions=(
            "You generate cautious research-use MRI analysis reports from "
            "structured model outputs. Use only the provided findings. Do not "
            "add diagnosis, tumor size, anatomical location, stage, treatment "
            "advice, prognosis, or any clinical detail not present in the input."
        ),
        input=formatted_prompt,
    )

    return response.output_text.strip()


def build_report(findings, use_llm=True):

    llm_enabled = (
        use_llm and
        os.getenv("MRI_REPORT_USE_LLM", "1").lower() not in {"0", "false", "no"}
    )

    if llm_enabled:
        try:
            return build_llm_report(findings)
        except Exception as exc:
            print(f"LLM report generation unavailable: {exc}")

    return build_template_report(findings)


def save_report(report_text, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as file:
        file.write(report_text)


if __name__ == "__main__":

    from findings_generator import generate_findings

    image_path = ("data/Testing/notumor/Te-no_10.jpg")

    findings = generate_findings(image_path)

    report = build_report(findings)

    output_path = ("outputs/reports/mri_report.md")

    save_report(report,output_path)

    print(report)

    print(f"\nSaved report to: {output_path}")
