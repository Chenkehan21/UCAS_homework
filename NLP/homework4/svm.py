from NB import load_data
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from NB import plot_confusion_matrix
import time


def svm():
    sentences, labels = load_data(path='./NLP_TC/traindata.txt')
    test_sentences, test_labels = load_data(path="./NLP_TC/testdata.txt")
    tfidf = TfidfVectorizer()
    train_data = tfidf.fit_transform(sentences)
    test_data = tfidf.transform(test_sentences)
    
    t1 = time.time()
    clf = SVC(C=5, kernel='rbf')
    clf.fit(train_data, labels)
    t2 = time.time()
    print("svm time: ", t2 - t1)
    predict = clf.predict(test_data)
    acc = metrics.classification_report(test_labels, predict)
    confusion = metrics.confusion_matrix(test_labels, predict)
    print(acc)
    print(confusion)
    plot_confusion_matrix(cm=confusion, target_names=list(set(labels)), normalize=False, title="SVM Confusion Matrix")
    

if __name__ == "__main__":
    svm()