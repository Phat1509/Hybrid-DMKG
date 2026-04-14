import json
import spacy
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
# 加载英文模型
nlp = spacy.load("en_core_web_sm")

def find_label_indices(question_tokens, label_tokens):
    """在question_tokens中查找label_tokens的起止索引"""
    for i in range(len(question_tokens) - len(label_tokens) + 1):
        if [t.lower() for t in question_tokens[i:i+len(label_tokens)]] == [t.lower() for t in label_tokens]:
            return i, i + len(label_tokens)
    return None, None

def get_synonyms(word):
    """获取单词的同义词集合（小写）"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def sequence_labeling(question, relation_label):
    doc = nlp(question)
    tokens = [token.text for token in doc]
    labels = ["O"] * len(tokens)
    rel_doc = nlp(relation_label)
    rel_lemmas = set(token.lemma_.lower() for token in rel_doc)
    rel_synonyms = set()
    for lemma in rel_lemmas:
        rel_synonyms |= get_synonyms(lemma)
    rel_all = rel_lemmas | rel_synonyms

    stemmer = PorterStemmer()
    rel_stems = set(stemmer.stem(word) for word in rel_all)

    for i, token in enumerate(doc):
        token_lemma = token.lemma_.lower()
        token_stem = stemmer.stem(token_lemma)
        # 跳过介词（ADP）
        if token.pos_ == "ADP":
            continue
        if token_lemma in rel_all or token_stem in rel_stems:
            labels[i] = "REL"
    return list(zip(tokens, labels))

def main():
    with open("./final_train_data.json", "r") as f:
        data = json.load(f)
    output = []
    for index, item in enumerate(data):
        print(index)
        question = item["question"]
        relation_label = item["relation_labels"]
        question = re.sub(r'(\w)(\?)$', r'\1 ?', question)
        seq_labels = sequence_labeling(question, relation_label)
        labels = [l for t, l in seq_labels]
        # 如果全是"O"，则跳过
        if all(l == "O" for l in labels):
            continue
        output.append({
            "question": question,
            "tokens": [t for t, l in seq_labels],
            "labels": labels
        })
    with open("nre_train.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()