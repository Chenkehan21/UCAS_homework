from NB import load_data
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


def svm():
    sentences, labels = load_data(path='./NLP_TC/traindata.txt')
    test_sentences, test_labels = load_data(path="./NLP_TC/testdata.txt")
    tfidf = TfidfVectorizer()
    train_data = tfidf.fit_transform(sentences)
    test_data = tfidf.transform(test_sentences)
    
    clf = SVC(C=5, kernel='rbf')
    clf.fit(train_data, labels)
    predict = clf.predict(test_data)
    acc = classification_report(test_labels, predict)
    print(acc)
    
    

if __name__ == "__main__":
    svm()