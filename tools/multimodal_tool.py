from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoImageProcessor,
    AutoModelForImageClassification,
    M2M100Tokenizer,
    M2M100ForConditionalGeneration
)
from PIL import Image
from config.config import translation_model_path, image_model_path
from tools.sentiment_tool import analyze_sentiment

def analyze_multimodal_content(text=None, file_path=None, translate_source_lang="en", translate_target_lang="fr"):
    """
    Analyze text sentiment, classify images, and translate text.
    """
    results = {}
    
    # Sentiment Analysis
    if text:
        sentiment_result = analyze_sentiment(text)
        results["sentiment_analysis"] = sentiment_result
    
    # Image Classification
    if file_path:
        try:
            processor = AutoImageProcessor.from_pretrained(image_model_path)
            model = AutoModelForImageClassification.from_pretrained(image_model_path)
            image = Image.open(file_path)
            image_pipeline = pipeline("image-classification", model=model, feature_extractor=processor)
            image_result = image_pipeline(image)
            results["image_classification"] = image_result
        except Exception as e:
            results["image_classification_error"] = str(e)
    
    # Text Translation
    if text and translate_source_lang and translate_target_lang:
        try:
            tokenizer = M2M100Tokenizer.from_pretrained(translation_model_path)
            model = M2M100ForConditionalGeneration.from_pretrained(translation_model_path)
            tokenizer.src_lang = translate_source_lang
            encoded_text = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(translate_target_lang))
            translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            results["translation"] = translated_text
        except Exception as e:
            results["translation_error"] = str(e)
    
    return results

# Tool definition for OpenAI function calling with updated parameters
multimodal_tool = {
    "type": "function",
    "function": {
        "name": "analyze_multimodal_content",
        "description": "tool can Analyze text sentiment, image classification, and translate text to target language only",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze for sentiment or translate."
                },
                "file_path": {
                    "type": "string",
                    "description": "The path to the image file for classification."
                },
                "translate_source_lang": {
                    "type": "string",
                    "description": "The source language for translation (for ex-'en')",
                    "default": "en"
                },
                "translate_target_lang": {
                    "type": "string",
                    "description": "The target language for translation (for ex-'fr')",
                    "default": "fr"
                }
            },
            "required": []
        }
    }
}