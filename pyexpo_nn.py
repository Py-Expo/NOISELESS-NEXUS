import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Sample HR provided keywords
hr_keywords = ["team player", "communication skills", "problem-solving", "leadership", "detail-oriented"]

# Sample resumes (you can replace this with your actual resume data)
resumes = [
    "Experienced team player with excellent communication skills and strong problem-solving abilities.",
    "Effective communicator with leadership qualities, adept at problem-solving and detail-oriented.",
    "Strong leadership skills, excellent communication, and a team player.",
    "Detail-oriented individual with a focus on problem-solving and effective communication."
]

def preprocess_text(text):
    """Preprocess text by removing stop words and lemmatization."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

# Preprocess HR keywords
hr_keywords_preprocessed = [preprocess_text(keyword) for keyword in hr_keywords]

# Preprocess resumes
resumes_preprocessed = [preprocess_text(resume) for resume in resumes]

# Create bag of words representations for HR keywords and resumes
vectorizer = CountVectorizer()
hr_keywords_bow = vectorizer.fit_transform(hr_keywords_preprocessed)
resumes_bow = vectorizer.transform(resumes_preprocessed)

# Calculate cosine similarity between HR keywords and resumes
similarity_scores = cosine_similarity(hr_keywords_bow, resumes_bow)

# Threshold and range for considering a resume as near the threshold
threshold = 0.5
threshold_range = 0.1  # Adjust this range as needed

# Check if any of the resumes have similarity score near the threshold
near_threshold_resumes = []
for j, scores in enumerate(similarity_scores.T):  # Transpose similarity scores to match resumes
    max_score = max(scores)
    if threshold - threshold_range <= max_score <= threshold + threshold_range:
        near_threshold_resumes.append((resumes[j], max_score))

# Print resumes near the threshold
if near_threshold_resumes:
    print("Resumes near the threshold:")
    for resume, score in near_threshold_resumes:
        print(f"Resume: {resume}\nSimilarity score: {score}\n")
else:
    print("No resumes are near the threshold.")