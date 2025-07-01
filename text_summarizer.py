import nltk
import heapq
import re

# Download required resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def clean_text(text):
    """Clean text by removing citations and extra whitespace."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', '', text)
    return text

def get_word_frequencies(words, stop_words):
    """Compute word frequencies excluding stopwords."""
    word_frequencies = {}
    for word in words:
        if word.isalnum() and word not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
    return word_frequencies

def score_sentences(sentences, word_frequencies, max_sentence_length=30):
    """Score sentences based on word frequency."""
    sentence_scores = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        if len(words) <= max_sentence_length:
            for word in words:
                if word in word_frequencies:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]
    return sentence_scores

def summarize(text, max_sentences=3):
    """Summarize the input text into a specified number of key sentences."""
    text = clean_text(text)
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    word_frequencies = get_word_frequencies(words, stop_words)
    sentence_scores = score_sentences(sentences, word_frequencies)
    summary_sentences = heapq.nlargest(max_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary
if __name__ == "__main__":
    input_text = """
    Artificial Intelligence (AI) has significantly transformed the way we interact with technology.
    From smart assistants and chatbots to recommendation engines and self-driving cars, AI is embedded in our daily lives.
    Natural Language Processing (NLP) is a subfield of AI that enables machines to understand and process human language.
    Applications of NLP include language translation, sentiment analysis, summarization, and more.
    This tool demonstrates how lengthy articles or passages can be summarized into concise and meaningful content.
    """

    print("Original Text:\n", input_text.strip())
    summary = summarize(input_text, max_sentences=2)
    print("\nSummarized Text:\n", summary)
