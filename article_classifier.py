"""Tag each publication with the best-matching topic from its author's research interests.

For every file in researcher_corpora/, the researcher's Google Scholar
"Research Interests" are split into candidate topics. Each article
(title + description) is scored against every topic with a keyword-overlap
heuristic, and the best topic above THRESHOLD is written back to the file in
the Classified_Topic / Classification_Confidence columns.

The script is idempotent: re-running it recomputes every label from scratch.
"""
import glob
import os
import re
import sys

import pandas as pd

CORPUS_DIR = "researcher_corpora"
THRESHOLD = 0.05
UNCLASSIFIED = "Unclassified"
MIN_WORD_LEN = 4  # shorter words ("a", "of", "and") are ignored for partial matching
STOP_WORDS = {'and', 'or', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}


def preprocess_text(text):
    """Lowercase, replace punctuation with spaces and collapse whitespace.

    Uses \\w so accented (French) and Arabic letters are kept.
    """
    if pd.isna(text):
        return ""
    text = re.sub(r'[^\w\s]|_', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', text).strip()


def extract_topics(interests):
    """Split a research-interests string into topic labels.

    Separators: , ; | & . newlines, and hyphens used as separators
    ("A - B", "A- B", "Business-Accounting") but not inside terms ("real-time").
    """
    if pd.isna(interests):
        return []
    interests = re.sub(r'\S+@\S+', ' ', str(interests))  # drop e-mail addresses
    topics = []
    for keyword in re.split(r'[,;|&.\n]+|\s-|-\s|(?<=[a-z])-(?=[A-Z])', interests):
        keyword = keyword.strip(" -.").lower()
        if len(keyword) > 2 and keyword not in STOP_WORDS and keyword not in topics:
            topics.append(keyword)
    return topics


def words_match(topic_word, article_word):
    """True for word variants such as cardiology/cardiological or pharmacy/pharmacies."""
    if topic_word in article_word or article_word in topic_word:
        return True
    shared = len(os.path.commonprefix([topic_word, article_word]))
    return shared >= max(5, int(0.75 * min(len(topic_word), len(article_word))))


def score_topic(article_text, article_words, topic):
    """Similarity in [0, 1] between a preprocessed article and a preprocessed topic."""
    topic_words = set(topic.split())
    if not article_words or not topic_words:
        return 0.0

    # 1. Jaccard word overlap
    word_overlap = len(article_words & topic_words) / len(article_words | topic_words)

    # 2. Share of meaningful topic words that (partially) appear in the article
    meaningful_topic = [w for w in topic_words if len(w) >= MIN_WORD_LEN]
    meaningful_article = [w for w in article_words if len(w) >= MIN_WORD_LEN]
    if meaningful_topic:
        matches = sum(
            any(words_match(t, a) for a in meaningful_article) for t in meaningful_topic
        )
        keyword_score = matches / len(meaningful_topic)
    else:
        keyword_score = 0.0

    # 3. Whole topic phrase appears in the article
    phrase_score = 1.0 if f" {topic} " in f" {article_text} " else 0.0

    return min(word_overlap * 0.4 + keyword_score * 0.4 + phrase_score * 0.2, 1.0)


def classify_article(title, description, topics):
    """Return (topic_label, confidence) for one article."""
    article_text = preprocess_text(f"{'' if pd.isna(title) else title} "
                                   f"{'' if pd.isna(description) else description}")
    if len(article_text) < 10:
        return UNCLASSIFIED, 0.0

    article_words = set(article_text.split())
    best_topic, best_score = UNCLASSIFIED, 0.0
    for label, normalized in topics:
        score = score_topic(article_text, article_words, normalized)
        if score > best_score and score >= THRESHOLD:
            best_topic, best_score = label, score
    return best_topic, round(best_score, 3)


def classify_file(path):
    """Classify every article in one researcher file and save it in place."""
    df = pd.read_csv(path)
    interests = df['Research Interests'].dropna()
    labels = extract_topics(interests.iloc[0]) if not interests.empty else []
    topics = [(label, preprocess_text(label)) for label in labels]

    results = [classify_article(row.get('Title'), row.get('Description'), topics)
               for _, row in df.iterrows()]
    df['Classified_Topic'] = [topic for topic, _ in results]
    df['Classification_Confidence'] = [score for _, score in results]

    df.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')
    return df, bool(topics)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.csv")))
    if not files:
        sys.exit(f"No CSV files found in {CORPUS_DIR}/")

    all_articles = []
    no_interests = []
    for path in files:
        df, has_topics = classify_file(path)
        name = df['Researcher Name'].iloc[0] if not df.empty else os.path.basename(path)
        if not has_topics:
            no_interests.append(name)
        all_articles.append(df[['Researcher Name', 'Classified_Topic', 'Classification_Confidence']])

    articles = pd.concat(all_articles, ignore_index=True)
    classified = articles[articles['Classified_Topic'] != UNCLASSIFIED]

    print(f"Researchers processed: {len(files)}")
    print(f"Researchers without research interests (all articles Unclassified): {len(no_interests)}")
    print(f"Articles: {len(articles)} | classified: {len(classified)} "
          f"({len(classified) / len(articles):.1%}) | unclassified: {len(articles) - len(classified)}")
    if not classified.empty:
        print(f"Average confidence of classified articles: "
              f"{classified['Classification_Confidence'].mean():.3f}")
        print("\nTop 10 topics:")
        for topic, count in classified['Classified_Topic'].value_counts().head(10).items():
            print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()
