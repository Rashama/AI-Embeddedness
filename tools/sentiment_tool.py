from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from config.config import sentiment_model_path

def analyze_sentiment(text,file_path=None):
    """
    Analyze the sentiment of the input text using the 'cardiffnlp/twitter-roberta-base-sentiment' model.
    Returns a sentiment score and label.
    """
    # Load the model and tokenizer from the local path
    tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
    
    # Create a sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    # Get the sentiment result
    result = sentiment_pipeline(text)[0]
    
    # Map the label to a sentiment score (-1 to 1)
    label_to_score = {
        "LABEL_0": -1,  # Negative
        "LABEL_1": 0,   # Neutral
        "LABEL_2": 1    # Positive
    }

    sentiment = {
        "LABEL_0": 'Negative',  # Negative
        "LABEL_1": 'Neutral',   # Neutral
        "LABEL_2": 'Positive'    # Positive
    }
    
    sentiment_score = label_to_score.get(result["label"], 0)
    sentiment_label = sentiment.get(result["label"],'Neutral')
    
    return {
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label
    }

# Tool definition for OpenAI function calling
sentiment_tool = {
    "type": "function",
    "function": {
        "name": "analyze_sentiment",
        "description": "Analyze the sentiment of a given text and return the sentiment score and label.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze for sentiment."
                }
            },
            "required": ["text"]
        }
    }
}