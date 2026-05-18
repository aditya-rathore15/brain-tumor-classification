def analyze_confidence(confidence_score):

    if confidence_score >= 0.90:
        confidence_level = "high"

        review_recommendation = (
            "Standard review recommended."
        )

    elif confidence_score >= 0.70:
        confidence_level = "moderate"

        review_recommendation = (
            "Prediction confidence is moderate. "
            "Careful specialist review is advised."
        )

    else:
        confidence_level = "low"

        review_recommendation = (
            "Prediction confidence is low. "
            "Strong specialist review is recommended."
        )

    return {
        "confidence_score": round(confidence_score, 4),
        "confidence_level": confidence_level,
        "review_recommendation": review_recommendation
    }


if __name__ == "__main__":

    result = analyze_confidence(0.8426)

    print(result)