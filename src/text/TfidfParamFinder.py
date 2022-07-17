from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from Constants import RANDOM_STATE
from spio.TextLoading import load_custom_dataset
from text.TextPreProcessing import preprocess_dataset

if __name__ == '__main__':
    dataframe = load_custom_dataset()

    dataset = preprocess_dataset(dataframe.Text.values)
    labels = dataframe.Score.values

    X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                        labels,
                                                        test_size=0.30,
                                                        random_state=RANDOM_STATE)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=set(stopwords.words("english")))),
        ('clf', OneVsOneClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
    ])
    parameters = {
        'tfidf__max_features': (2000, 3000, 5000),
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__estimator__alpha': (1e-2, 1e-3)
    }

    grid_search = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)

    print("Best parameters set:", grid_search.best_estimator_.steps)
