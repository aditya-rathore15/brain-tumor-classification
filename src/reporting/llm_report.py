import os

from findings_generator import generate_findings


PROMPT_PATH = ("src/reporting/prompts/report_prompt.txt")


def load_prompt_template():

    with open(PROMPT_PATH, "r") as file:
        return file.read()


def build_report(findings):

    prompt_template = load_prompt_template()

    formatted_prompt = prompt_template.format(
        predicted_class=findings["predicted_class"],
        confidence_score=findings["confidence_score"],
        confidence_level=findings["confidence_level"],
        attention_quality=findings["attention_quality"],
        interpretation_summary=findings["interpretation_summary"],
        review_recommendation=findings["review_recommendation"],
        limitations="\n- " + "\n- ".join(findings["limitations"])
    )

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


def save_report(report_text, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as file:
        file.write(report_text)


if __name__ == "__main__":

    image_path = ("data/Testing/notumor/Te-no_10.jpg")

    findings = generate_findings(image_path)

    report = build_report(findings)

    output_path = ("outputs/reports/mri_report.md")

    save_report(report,output_path)

    print(report)

    print(f"\nSaved report to: {output_path}")