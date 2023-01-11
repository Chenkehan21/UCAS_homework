from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def load_data(path='./NLP_TC/traindata.txt'):
    with open(path, 'r') as f:
        text = f.readlines()
    
    sentences = []
    labels = []
    for line in text:
        line = line.split('\t')
        label = line[0]
        content = line[1].strip()
        sentences.append(content)
        labels.append(label)
        
    print("trainset size: ", len(text))
    
    return sentences, labels



def Naiive_Bayes():
    sentences, labels = load_data()
    tfidf = TfidfVectorizer()
    train_data = tfidf.fit_transform(sentences)
    
    nb = MultinomialNB()
    nb.fit(train_data, labels)
    
    test_path="./NLP_TC/testdata.txt"
    sentences, labels = load_data(test_path)
    test_data = tfidf.transform(sentences)
    predict = nb.predict(test_data)
    acc = classification_report(labels, predict)
    print(acc)
    

if __name__ == "__main__":
    Naiive_Bayes()