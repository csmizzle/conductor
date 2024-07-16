from transformers import pipeline
import os

# gibberish detector model
selected_model = "madhurjindal/autonlp-Gibberish-Detector-492513457"


def gibberish_detector(text: str):
    # load the model
    classifier = pipeline(
        "text-classification",
        model=selected_model,
        device=os.getenv("GPU_DEVICE", None),
    )
    # chunk the text into multiple 512 token chunks
    chunks = []
    for i in range(0, len(text), 512):
        chunks.append(text[i : i + 512])
    # classify each chunk
    results = [classifier(chunk) for chunk in chunks]
    return results


def gibberish_detector_chunks(text: str):
    # classify the text
    results = gibberish_detector(text)
    # check if any of the chunks are gibberish
    detections = {}
    # get average for each label: noise, word_salad, mild_gibberish, clean
    for result in results:
        for label in result:
            if label["label"] not in detections:
                detections[label["label"]] = []
            detections[label["label"]].append(label["score"])
    # get average for each label
    for key, value in detections.items():
        detections[key] = sum(value) / len(value)
    return detections


def gibberish(
    text,
    gibberish_groups: list = ["noise", "word salad", "mild gibberish"],
) -> dict:
    """
    Determine if the text is gibberish
    """
    gibberish_average = 0
    gibberish_scores = []
    # run gibberish pipeline
    gibberish_data = gibberish_detector_chunks(text)
    # group the gibberish groups based on input
    for group in gibberish_groups:
        if group in gibberish_data:
            gibberish_scores.append(gibberish_data[group])
    # get average of the gibberish groups
    if len(gibberish_scores) > 0:
        gibberish_average = sum(gibberish_scores) / len(gibberish_scores)
    # get clean score if in gibberish data
    clean_score = gibberish_data.get("clean", 0)
    # return gibberish data
    return {
        "gibberish": gibberish_average,
        "clean": clean_score,
    }


def is_gibberish(
    text,
    gibberish_groups: list = ["noise", "word salad", "mild gibberish"],
    gibberish_threshold: float = 0.5,
) -> bool:
    """
    Filter out gibberish text
    """
    # get gibberish data
    gibberish_data = gibberish(text, gibberish_groups)
    # check if gibberish is above threshold
    if gibberish_data["gibberish"] > gibberish_threshold:
        return True
    return False
