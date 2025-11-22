import nltk
from textblob import TextBlob
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    """Tokenize text using NLTK."""
    tokens = nltk.word_tokenize(text)
    return tokens

def analyze_sentiment(text):
    """Sentiment analysis using TextBlob."""
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

def named_entity_recognition(text):
    """Perform NER using spaCy."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def main():
    print("\n-----------------------------------")
    print("NATURAL LANGUAGE PROCESSING DEMO")
    print("-----------------------------------")

    user_input = input("\nEnter a sentence: ")

    # 1. Tokenization
    tokens = tokenize_text(user_input)
    print("\n> Tokenized Words:")
    print(tokens)

    # 2. Sentiment Analysis
    polarity, subjectivity = analyze_sentiment(user_input)
    print("\n> Sentiment Analysis:")
    print(f"Polarity: {polarity}  (Range: -1 = negative, 1 = positive)")
    print(f"Subjectivity: {subjectivity}  (Range: 0 = facts, 1 = opinions)")

    # 3. Named Entity Recognition
    entities = named_entity_recognition(user_input)
    print("\n> Named Entities:")
    if len(entities) == 0:
        print("No entities detected.")
    else:
        for ent, label in entities:
            print(f"{ent} -> {label}")

    print("\n-----------------------------------")
    print("NLP ANALYSIS COMPLETE YAHOOO!")
    print("-----------------------------------\n")

if __name__ == "__main__":
    main()

