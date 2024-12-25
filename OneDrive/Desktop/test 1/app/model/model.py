from pathlib import Path
import dill 
from sklearn.base import BaseEstimator, TransformerMixin
import re 


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # This is a no-op for this transformer.
        return self

    def transform(self, X, y=None):
        # Transform the input data
        processed_X = []
        for text in X:
            if isinstance(text, bytes):
                text = text.decode('utf-8')  # Decode bytes to string if necessary
            text = re.sub(r'[!@#$(),\n"%^*?:;~`0-9]', ' ', text)  # Remove unwanted characters
            text = re.sub(r'[[]]', ' ', text)
            text = text.lower()
            processed_X.append(text)
        return processed_X
    
   
__version__ = "1.0.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = dill.load(f)


classes = [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]

# Prediction function
def predict_pipeline(input_text):
    # Pass input directly to the pipeline
    if isinstance(input_text, str):
        input_text = [input_text]

    # Predict using the pipeline
    predictions = model.predict(input_text)
    
    # Map the numerical prediction to the corresponding class
    predicted_language = classes[predictions[0]]
    return predicted_language

if __name__ == "__main__":
    sample_input = "Bonjour, ceci est un test!"
    try:
        print("Pipeline prediction:", predict_pipeline(sample_input))
    except Exception as e:
        print("Error:", e)