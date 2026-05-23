from inference import load_model, predict_image

from confidence import analyze_confidence

from gradcam_service import generate_gradcam


def generate_findings(image_path):

    model = load_model()

    prediction_result = predict_image(
        image_path,
        model
    )

    confidence_result = analyze_confidence(
        prediction_result["confidence"]
    )

    gradcam_result = generate_gradcam(
        image_path=image_path,
        model=model,
        confidence=prediction_result["confidence"]
    )

    findings = {

        "predicted_class":
            prediction_result["prediction"],

        "confidence_score":
            confidence_result["confidence_score"],

        "confidence_level":
            confidence_result["confidence_level"],

        "probabilities":
            prediction_result["probabilities"],

        "attention_quality":
            gradcam_result["attention_quality"],

        "gradcam_path":
            gradcam_result["gradcam_path"],

        "review_recommendation":
            confidence_result["review_recommendation"],

        "interpretation_summary":
            generate_interpretation_summary(
                prediction_result["prediction"],
                confidence_result["confidence_level"],
                gradcam_result["attention_quality"]
            ),

        "limitations": [
            "Research-use only",
            "Not intended for clinical diagnosis",
            "Requires specialist review"
        ]
    }

    return findings


def generate_interpretation_summary(
    prediction,
    confidence_level,
    attention_quality
):

    return (
        f"The model predicts {prediction} "
        f"with {confidence_level} confidence. "
        f"Grad-CAM analysis indicates "
        f"{attention_quality} attention patterns."
    )


if __name__ == "__main__":

    findings = generate_findings(
        "data/Testing/notumor/Te-no_10.jpg"
    )

    print(findings)