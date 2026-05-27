import os
import json
import argparse

from findings_generator import generate_findings

from llm_report import (build_report, save_report)


def save_prediction_json(findings, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as file:

        json.dump(findings, file, indent=4)


def main(image_path):

    print("\nGenerating findings...\n")

    findings = generate_findings(image_path)

    print("Generating report...\n")

    report = build_report(findings)

    report_output_path = ("outputs/reports/mri_report.md")

    json_output_path = ("outputs/reports/prediction.json")

    save_report(report, report_output_path)

    save_prediction_json(findings, json_output_path)

    print(report)

    print("\nArtifacts Generated:")

    print(f"- Report: {report_output_path}")

    print(f"- Prediction JSON: {json_output_path}")

    print(
        f"- Grad-CAM: "
        f"{findings['gradcam_path']}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, required=True, help="Path to MRI image")

    args = parser.parse_args()

    main(args.image)