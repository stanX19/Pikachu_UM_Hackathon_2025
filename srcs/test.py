from transformers import pipeline
import torch

class IntentPredictor:
    """Predicts intent from transcribed speech text using zero-shot classification."""

    # Define possible intent labels
    INTENTS = [
        "navigation",
        "accept_order",
        "chat_passenger",
        "i_have_fetched_passenger",
        "exit_voice_mode"
    ]

    # Confidence threshold for accepting a label
    CONFIDENCE_THRESHOLD = 0.6  # adjust as needed

    # Use GPU if available
    DEVICE = 0 if torch.cuda.is_available() else -1

    # Initialize zero-shot classifier with distilbert model
    classifier = pipeline(
        "zero-shot-classification",
        model="distilbert-base-uncased",
        device=DEVICE
    )

    @classmethod
    def predict_intent(cls, text: str) -> str:
        """
        Predict the intent from the transcribed text using a zero-shot classifier.

        Args:
            text: The transcribed text from speech recognition

        Returns:
            The predicted intent as a string, or 'unknown' if confidence is too low
        """
        result = cls.classifier(text, cls.INTENTS)
        top_label = result["labels"][0]  # Top intent label
        top_score = result["scores"][0]  # Confidence score

        # If confidence is too low, fallback to unknown
        if top_score < cls.CONFIDENCE_THRESHOLD:
            return "unknown"

        return top_label

# Example usage
if __name__ == "__main__":
    examples = [
        "Can you take me downtown?",  # Likely navigation
        "Accept the ride request",    # Likely accept_order
        "Say hi to the customer",     # Likely chat_passenger
        "I have the passenger in my car",  # Likely i_have_fetched_passenger
        "End voice mode now",         # Likely exit_voice_mode
        "Open the sunroof"             # Likely unknown
    ]

    for example in examples:
        intent = IntentPredictor.predict_intent(example)
        print(f"Text: '{example}' → Predicted Intent: '{intent}'")
