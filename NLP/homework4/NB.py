from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import itertools
import time


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
    t1 = time.time()
    sentences, labels = load_data()
    tfidf = TfidfVectorizer()
    train_data = tfidf.fit_transform(sentences)
    t2 = time.time()
    print("date preprocess time: ", t2 - t1)
    
    t3 = time.time()
    nb = MultinomialNB()
    nb.fit(train_data, labels)
    t4 = time.time()
    print("NB time: ", t4 - t3)
    
    test_path="./NLP_TC/testdata.txt"
    sentences, labels = load_data(test_path)
    test_data = tfidf.transform(sentences)
    predict = nb.predict(test_data)
    acc = metrics.classification_report(labels, predict)
    confusion = metrics.confusion_matrix(labels, predict)
    print(acc)
    print(confusion)
    plot_confusion_matrix(cm=confusion, target_names=list(set(labels)), normalize=False, title="NB Confusion Matrix")
    

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
	#plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
    plt.show()
    


if __name__ == "__main__":
    Naiive_Bayes()