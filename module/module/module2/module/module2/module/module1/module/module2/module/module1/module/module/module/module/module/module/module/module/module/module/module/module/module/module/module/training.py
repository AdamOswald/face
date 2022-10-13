{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Sentiment Analysis with XGBoost on Algorithmia"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Train, evaluate and test XGBoost model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split, RandomizedSearchCV\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "from string import punctuation\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "from xgboost import XGBClassifier\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load the training data\n",
    "Let's load our training data, take a look at a few rows and one of the review texts in detail."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"./data/amazon_musical_reviews/Musical_instruments_reviews.csv\")\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"reviewText\"].iloc[1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Preprocessing\n",
    "Time to process our texts! Basically, we'll:\n",
    "- Remove the English stopwords\n",
    "- Remove punctuations\n",
    "- Drop unused columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import nltk\n",
    "nltk.download('stopwords')\n",
    "\n",
    "def threshold_ratings(data):\n",
    "    def threshold_overall_rating(rating):\n",
    "        return 0 if int(rating)<=3 else 1\n",
    "    data[\"overall\"] = data[\"overall\"].apply(threshold_overall_rating)\n",
    "\n",
    "def remove_stopwords_punctuation(data):\n",
    "    data[\"review\"] = data[\"reviewText\"] + data[\"summary\"]\n",
    "\n",
    "    puncs = list(punctuation)\n",
    "    stops = stopwords.words(\"english\")\n",
    "\n",
    "    def remove_stopwords_in_str(input_str):\n",
    "        filtered = [char for char in str(input_str).split() if char not in stops]\n",
    "        return ' '.join(filtered)\n",
    "\n",
    "    def remove_punc_in_str(input_str):\n",
    "        filtered = [char for char in input_str if char not in puncs]\n",
    "        return ''.join(filtered)\n",
    "\n",
    "    def remove_stopwords_in_series(input_series):\n",
    "        text_clean = []\n",
    "        for i in range(len(input_series)):\n",
    "            text_clean.append(remove_stopwords_in_str(input_series[i]))\n",
    "        return text_clean\n",
    "\n",
    "    def remove_punc_in_series(input_series):\n",
    "        text_clean = []\n",
    "        for i in range(len(input_series)):\n",
    "            text_clean.append(remove_punc_in_str(input_series[i]))\n",
    "        return text_clean\n",
    "\n",
    "    data[\"review\"] = remove_stopwords_in_series(data[\"review\"].str.lower())\n",
    "    data[\"review\"] = remove_punc_in_series(data[\"review\"].str.lower())\n",
    "\n",
    "def drop_unused_colums(data):\n",
    "    data.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime', \"reviewText\", \"summary\"], axis=1, inplace=True)\n",
    "\n",
    "def preprocess_reviews(data):\n",
    "    remove_stopwords_punctuation(data)\n",
    "    threshold_ratings(data)\n",
    "    drop_unused_colums(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "preprocess_reviews(data)\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Split our training and test sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rand_seed = 42\n",
    "X = data[\"review\"]\n",
    "y = data[\"overall\"]\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Mini randomized search\n",
    "Let's set up a very basic cross-validated randomized search over parameter settings."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "params = {\"max_depth\": range(9,12), \"min_child_weight\": range(5,8)}\n",
    "rand_search_cv = RandomizedSearchCV(XGBClassifier(), param_distributions=params, n_iter=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Pipeline to vectorize, transform and fit\n",
    "Time to vectorize our data, transform it and then fit our model to it.\n",
    "To be able to feed the text data as numeric values to our model, we will first convert our texts into a matrix of token counts using a CountVectorizer. Then we will convert the count matrix to a normalized tf-idf (term-frequency times inverse document-frequency) representation. Using this transformer, we will be scaling down the impact of tokens that occur very frequently, because they convey less information to us. On the contrary, we will be scaling up the impact of the tokens that occur in a small fraction of the training data because they are more informative to us."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "model  = Pipeline([\n",
    "    ('vect', CountVectorizer()),\n",
    "    ('tfidf', TfidfTransformer()),\n",
    "    ('model', rand_search_cv)\n",
    "])\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Predict and calculate accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "predictions = model.predict(X_test)\n",
    "acc = accuracy_score(y_test, predictions)\n",
    "print(f\"Model Accuracy: {round(acc * 100, 2)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Save XGBoost model to a file\n",
    "We should save the created model object to a local path. The Github action will then take the model object from this path and upload it to Algorithmia."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "joblib.dump(model, \"model.pkl\", compress=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Test our serving (Algorithm) code\n",
    "\n",
    "We will now test our algorithm code, ie. **`xgboost_automated_github.py`** script, by simply executing it with the `%run` macro. The script will begin its execution through our `if __name__ == \"__main__\"` line. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "output_type": "error",
     "ename": "DataApiError",
     "evalue": "unable to get file asli/xgboost_automated_github/model_1efcc5e311207268e450d6018debe7012f182e71.pkl - authorization required",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mDataApiError\u001b[0m                              Traceback (most recent call last)",
      "\u001b[0;32m~/repos/algorithmia_ci/demo_autodeploy_algo_on_github/xgboost_automated_github.py\u001b[0m in \u001b[0;36m&lt;module&gt;\u001b[0;34m\u001b[0m\n\u001b[1;32m     58\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     59\u001b[0m \u001b[0mconfig\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mload_model_config\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---&gt; 60\u001b[0;31m \u001b[0mxgb_path\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mxgb_obj\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mload_model\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mconfig\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     61\u001b[0m \u001b[0massert_model_md5\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mxgb_path\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     62\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/repos/algorithmia_ci/demo_autodeploy_algo_on_github/xgboost_automated_github.py\u001b[0m in \u001b[0;36mload_model\u001b[0;34m(config)\u001b[0m\n\u001b[1;32m     35\u001b[0m     \u001b[0;34m&quot;&quot;&quot;Loads the model object from the file at model_filepath key in config dict&quot;&quot;&quot;\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     36\u001b[0m     \u001b[0mmodel_path\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mconfig\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m&quot;model_filepath&quot;\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---&gt; 37\u001b[0;31m     \u001b[0mmodel_file\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mclient\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfile\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel_path\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgetFile\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     38\u001b[0m     \u001b[0mmodel_obj\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mjoblib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mload\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel_file\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     39\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mmodel_file\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmodel_obj\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.8/site-packages/Algorithmia/datafile.py\u001b[0m in \u001b[0;36mgetFile\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m     35\u001b[0m         \u001b[0mexists\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0merror\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexistsWithError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     36\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mexists\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---&gt; 37\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mDataApiError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m&#39;unable to get file {} - {}&#39;\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0merror\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     38\u001b[0m         \u001b[0;31m# Make HTTP get request\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     39\u001b[0m         \u001b[0mresponse\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclient\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgetHelper\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0murl\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mDataApiError\u001b[0m: unable to get file asli/xgboost_automated_github/model_1efcc5e311207268e450d6018debe7012f182e71.pkl - authorization required"
     ]
    }
   ],
   "source": [
    "%run xgboost_automated_github/src/xgboost_automated_github.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Final checks before committing this notebook\n",
    "\n",
    "- Make sure `xgboost_automated_github.py` file is in good shape to be pushed to Algorithmia.\n",
    "\n",
    "- Make sure Algorithmia API Key is not committed and pushed in any file!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3-final"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}


Sentiment Analysis with XGBoost on Algorithmia
1. Train, evaluate and test XGBoost model
In [ ]:
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from string import punctuation
from nltk.corpus import stopwords

from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib
Load the training data

Let's load our training data, take a look at a few rows and one of the review texts in detail.

In [ ]:
data = pd.read_csv("./data/amazon_musical_reviews/Musical_instruments_reviews.csv")
data.head()
In [ ]:
data["reviewText"].iloc[1]
Preprocessing

Time to process our texts! Basically, we'll:

Remove the English stopwords
Remove punctuations
Drop unused columns
In [ ]:
import nltk
nltk.download('stopwords')

def threshold_ratings(data):
    def threshold_overall_rating(rating):
        return 0 if int(rating)<=3 else 1
    data["overall"] = data["overall"].apply(threshold_overall_rating)

def remove_stopwords_punctuation(data):
    data["review"] = data["reviewText"] + data["summary"]

    puncs = list(punctuation)
    stops = stopwords.words("english")

    def remove_stopwords_in_str(input_str):
        filtered = [char for char in str(input_str).split() if char not in stops]
        return ' '.join(filtered)

    def remove_punc_in_str(input_str):
        filtered = [char for char in input_str if char not in puncs]
        return ''.join(filtered)

    def remove_stopwords_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_stopwords_in_str(input_series[i]))
        return text_clean

    def remove_punc_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_punc_in_str(input_series[i]))
        return text_clean

    data["review"] = remove_stopwords_in_series(data["review"].str.lower())
    data["review"] = remove_punc_in_series(data["review"].str.lower())

def drop_unused_colums(data):
    data.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime', "reviewText", "summary"], axis=1, inplace=True)

def preprocess_reviews(data):
    remove_stopwords_punctuation(data)
    threshold_ratings(data)
    drop_unused_colums(data)
In [ ]:
preprocess_reviews(data)
data.head()
Split our training and test sets
In [ ]:
rand_seed = 42
X = data["review"]
y = data["overall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)
Mini randomized search

Let's set up a very basic cross-validated randomized search over parameter settings.

In [ ]:
params = {"max_depth": range(9,12), "min_child_weight": range(5,8)}
rand_search_cv = RandomizedSearchCV(XGBClassifier(), param_distributions=params, n_iter=1)
Pipeline to vectorize, transform and fit

Time to vectorize our data, transform it and then fit our model to it. To be able to feed the text data as numeric values to our model, we will first convert our texts into a matrix of token counts using a CountVectorizer. Then we will convert the count matrix to a normalized tf-idf (term-frequency times inverse document-frequency) representation. Using this transformer, we will be scaling down the impact of tokens that occur very frequently, because they convey less information to us. On the contrary, we will be scaling up the impact of the tokens that occur in a small fraction of the training data because they are more informative to us.

In [ ]:
model  = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', rand_search_cv)
])
model.fit(X_train, y_train)
Predict and calculate accuracy
In [ ]:
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {round(acc * 100, 2)}")
2. Save XGBoost model to a file

We should save the created model object to a local path. The Github action will then take the model object from this path and upload it to Algorithmia.

In [ ]:
joblib.dump(model, "model.pkl", compress=True)
3. Test our serving (Algorithm) code

We will now test our algorithm code, ie. xgboost_automated_github.py script, by simply executing it with the %run macro. The script will begin its execution through our if __name__ == "__main__" line.

In [1]:
%run xgboost_automated_github/src/xgboost_automated_github.py
---------------------------------------------------------------------------
DataApiError                              Traceback (most recent call last)
~/repos/algorithmia_ci/demo_autodeploy_algo_on_github/xgboost_automated_github.py in &lt;module&gt;
     58 
     59 config = load_model_config()
---&gt; 60 xgb_path, xgb_obj = load_model(config)
     61 assert_model_md5(xgb_path)
     62 

~/repos/algorithmia_ci/demo_autodeploy_algo_on_github/xgboost_automated_github.py in load_model(config)
     35     &quot;&quot;&quot;Loads the model object from the file at model_filepath key in config dict&quot;&quot;&quot;
     36     model_path = config[&quot;model_filepath&quot;]
---&gt; 37     model_file = client.file(model_path).getFile().name
     38     model_obj = joblib.load(model_file)
     39     return model_file, model_obj

~/opt/anaconda3/lib/python3.8/site-packages/Algorithmia/datafile.py in getFile(self)
     35         exists, error = self.existsWithError()
     36         if not exists:
---&gt; 37             raise DataApiError(&#39;unable to get file {} - {}&#39;.format(self.path, error))
     38         # Make HTTP get request
     39         response = self.client.getHelper(self.url)

DataApiError: unable to get file asli/xgboost_automated_github/model_1efcc5e311207268e450d6018debe7012f182e71.pkl - authorization required
Final checks before committing this notebook

Make sure xgboost_automated_github.py file is in good shape to be pushed to Algorithmia.

Make sure Algorithmia API Key is not committed and pushed in any file!

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from string import punctuation
from nltk.corpus import stopwords

from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib

data = pd.read_csv("./data/amazon_musical_reviews/Musical_instruments_reviews.csv")
data.head()

data["reviewText"].iloc[1]

import nltk
nltk.download('stopwords')

def threshold_ratings(data):
    def threshold_overall_rating(rating):
        return 0 if int(rating)<=3 else 1
    data["overall"] = data["overall"].apply(threshold_overall_rating)

def remove_stopwords_punctuation(data):
    data["review"] = data["reviewText"] + data["summary"]

    puncs = list(punctuation)
    stops = stopwords.words("english")

    def remove_stopwords_in_str(input_str):
        filtered = [char for char in str(input_str).split() if char not in stops]
        return ' '.join(filtered)

    def remove_punc_in_str(input_str):
        filtered = [char for char in input_str if char not in puncs]
        return ''.join(filtered)

    def remove_stopwords_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_stopwords_in_str(input_series[i]))
        return text_clean

    def remove_punc_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_punc_in_str(input_series[i]))
        return text_clean

    data["review"] = remove_stopwords_in_series(data["review"].str.lower())
    data["review"] = remove_punc_in_series(data["review"].str.lower())

def drop_unused_colums(data):
    data.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime', "reviewText", "summary"], axis=1, inplace=True)

def preprocess_reviews(data):
    remove_stopwords_punctuation(data)
    threshold_ratings(data)
    drop_unused_colums(data)

preprocess_reviews(data)
data.head()

rand_seed = 42
X = data["review"]
y = data["overall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)

params = {"max_depth": range(9,12), "min_child_weight": range(5,8)}
rand_search_cv = RandomizedSearchCV(XGBClassifier(), param_distributions=params, n_iter=1)

model  = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', rand_search_cv)
])
model.fit(X_train, y_train)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {round(acc * 100, 2)}")

joblib.dump(model, "model.pkl", compress=True)

%run xgboost_automated_github/src/xgboost_automated_github.py

---------------------------------------------------------------------------
DataApiError                              Traceback (most recent call last)
~/repos/algorithmia_ci/demo_autodeploy_algo_on_github/xgboost_automated_github.py in &lt;module&gt;
     58 
     59 config = load_model_config()
---&gt; 60 xgb_path, xgb_obj = load_model(config)
     61 assert_model_md5(xgb_path)
     62 

~/repos/algorithmia_ci/demo_autodeploy_algo_on_github/xgboost_automated_github.py in load_model(config)
     35     &quot;&quot;&quot;Loads the model object from the file at model_filepath key in config dict&quot;&quot;&quot;
     36     model_path = config[&quot;model_filepath&quot;]
---&gt; 37     model_file = client.file(model_path).getFile().name
     38     model_obj = joblib.load(model_file)
     39     return model_file, model_obj

~/opt/anaconda3/lib/python3.8/site-packages/Algorithmia/datafile.py in getFile(self)
     35         exists, error = self.existsWithError()
     36         if not exists:
---&gt; 37             raise DataApiError(&#39;unable to get file {} - {}&#39;.format(self.path, error))
     38         # Make HTTP get request
     39         response = self.client.getHelper(self.url)

DataApiError: unable to get file asli/xgboost_automated_github/model_1efcc5e311207268e450d6018debe7012f182e71.pkl - authorization required

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Sentiment Analysis with XGBoost on Algorithmia"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Train, evaluate and test XGBoost model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split, RandomizedSearchCV\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "from string import punctuation\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "from xgboost import XGBClassifier\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load the training data\n",
    "Let's load our training data, take a look at a few rows and one of the review texts in detail."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"./data/amazon_musical_reviews/Musical_instruments_reviews.csv\")\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"reviewText\"].iloc[1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Preprocessing\n",
    "Time to process our texts! Basically, we'll:\n",
    "- Remove the English stopwords\n",
    "- Remove punctuations\n",
    "- Drop unused columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import nltk\n",
    "nltk.download('stopwords')\n",
    "\n",
    "def threshold_ratings(data):\n",
    "    def threshold_overall_rating(rating):\n",
    "        return 0 if int(rating)<=3 else 1\n",
    "    data[\"overall\"] = data[\"overall\"].apply(threshold_overall_rating)\n",
    "\n",
    "def remove_stopwords_punctuation(data):\n",
    "    data[\"review\"] = data[\"reviewText\"] + data[\"summary\"]\n",
    "\n",
    "    puncs = list(punctuation)\n",
    "    stops = stopwords.words(\"english\")\n",
    "\n",
    "    def remove_stopwords_in_str(input_str):\n",
    "        filtered = [char for char in str(input_str).split() if char not in stops]\n",
    "        return ' '.join(filtered)\n",
    "\n",
    "    def remove_punc_in_str(input_str):\n",
    "        filtered = [char for char in input_str if char not in puncs]\n",
    "        return ''.join(filtered)\n",
    "\n",
    "    def remove_stopwords_in_series(input_series):\n",
    "        text_clean = []\n",
    "        for i in range(len(input_series)):\n",
    "            text_clean.append(remove_stopwords_in_str(input_series[i]))\n",
    "        return text_clean\n",
    "\n",
    "    def remove_punc_in_series(input_series):\n",
    "        text_clean = []\n",
    "        for i in range(len(input_series)):\n",
    "            text_clean.append(remove_punc_in_str(input_series[i]))\n",
    "        return text_clean\n",
    "\n",
    "    data[\"review\"] = remove_stopwords_in_series(data[\"review\"].str.lower())\n",
    "    data[\"review\"] = remove_punc_in_series(data[\"review\"].str.lower())\n",
    "\n",
    "def drop_unused_colums(data):\n",
    "    data.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime', \"reviewText\", \"summary\"], axis=1, inplace=True)\n",
    "\n",
    "def preprocess_reviews(data):\n",
    "    remove_stopwords_punctuation(data)\n",
    "    threshold_ratings(data)\n",
    "    drop_unused_colums(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "preprocess_reviews(data)\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Split our training and test sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rand_seed = 42\n",
    "X = data[\"review\"]\n",
    "y = data[\"overall\"]\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Mini randomized search\n",
    "Let's set up a very basic cross-validated randomized search over parameter settings."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "params = {\"max_depth\": range(9,12), \"min_child_weight\": range(5,8)}\n",
    "rand_search_cv = RandomizedSearchCV(XGBClassifier(), param_distributions=params, n_iter=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Pipeline to vectorize, transform and fit\n",
    "Time to vectorize our data, transform it and then fit our model to it.\n",
    "To be able to feed the text data as numeric values to our model, we will first convert our texts into a matrix of token counts using a CountVectorizer. Then we will convert the count matrix to a normalized tf-idf (term-frequency times inverse document-frequency) representation. Using this transformer, we will be scaling down the impact of tokens that occur very frequently, because they convey less information to us. On the contrary, we will be scaling up the impact of the tokens that occur in a small fraction of the training data because they are more informative to us."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "model  = Pipeline([\n",
    "    ('vect', CountVectorizer()),\n",
    "    ('tfidf', TfidfTransformer()),\n",
    "    ('model', rand_search_cv)\n",
    "])\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Predict and calculate accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "predictions = model.predict(X_test)\n",
    "acc = accuracy_score(y_test, predictions)\n",
    "print(f\"Model Accuracy: {round(acc * 100, 2)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Save XGBoost model to a file\n",
    "We should save the created model object to a local path. The Github action will then take the model object from this path and upload it to Algorithmia."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "joblib.dump(model, \"model.pkl\", compress=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Test our serving (Algorithm) code\n",
    "\n",
    "We will now test our algorithm code, ie. **`xgboost_automated.py`** script, by simply executing it with the `%run` macro. The script will begin its execution through our `if __name__ == \"__main__\"` line. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "%run xgboost_automated/src/xgboost_automated.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Final checks before committing this notebook\n",
    "\n",
    "- Make sure `xgboost_automated.py` file is in good shape to be pushed to Algorithmia.\n",
    "\n",
    "- Make sure Algorithmia API Key is not committed and pushed in any file!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3-final"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

Sentiment Analysis with XGBoost on Algorithmia
1. Train, evaluate and test XGBoost model
In [ ]:
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from string import punctuation
from nltk.corpus import stopwords

from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib
Load the training data

Let's load our training data, take a look at a few rows and one of the review texts in detail.

In [ ]:
data = pd.read_csv("./data/amazon_musical_reviews/Musical_instruments_reviews.csv")
data.head()
In [ ]:
data["reviewText"].iloc[1]
Preprocessing

Time to process our texts! Basically, we'll:

Remove the English stopwords
Remove punctuations
Drop unused columns
In [ ]:
import nltk
nltk.download('stopwords')

def threshold_ratings(data):
    def threshold_overall_rating(rating):
        return 0 if int(rating)<=3 else 1
    data["overall"] = data["overall"].apply(threshold_overall_rating)

def remove_stopwords_punctuation(data):
    data["review"] = data["reviewText"] + data["summary"]

    puncs = list(punctuation)
    stops = stopwords.words("english")

    def remove_stopwords_in_str(input_str):
        filtered = [char for char in str(input_str).split() if char not in stops]
        return ' '.join(filtered)

    def remove_punc_in_str(input_str):
        filtered = [char for char in input_str if char not in puncs]
        return ''.join(filtered)

    def remove_stopwords_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_stopwords_in_str(input_series[i]))
        return text_clean

    def remove_punc_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_punc_in_str(input_series[i]))
        return text_clean

    data["review"] = remove_stopwords_in_series(data["review"].str.lower())
    data["review"] = remove_punc_in_series(data["review"].str.lower())

def drop_unused_colums(data):
    data.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime', "reviewText", "summary"], axis=1, inplace=True)

def preprocess_reviews(data):
    remove_stopwords_punctuation(data)
    threshold_ratings(data)
    drop_unused_colums(data)
In [ ]:
preprocess_reviews(data)
data.head()
Split our training and test sets
In [ ]:
rand_seed = 42
X = data["review"]
y = data["overall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)
Mini randomized search

Let's set up a very basic cross-validated randomized search over parameter settings.

In [ ]:
params = {"max_depth": range(9,12), "min_child_weight": range(5,8)}
rand_search_cv = RandomizedSearchCV(XGBClassifier(), param_distributions=params, n_iter=1)
Pipeline to vectorize, transform and fit

Time to vectorize our data, transform it and then fit our model to it. To be able to feed the text data as numeric values to our model, we will first convert our texts into a matrix of token counts using a CountVectorizer. Then we will convert the count matrix to a normalized tf-idf (term-frequency times inverse document-frequency) representation. Using this transformer, we will be scaling down the impact of tokens that occur very frequently, because they convey less information to us. On the contrary, we will be scaling up the impact of the tokens that occur in a small fraction of the training data because they are more informative to us.

In [ ]:
model  = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', rand_search_cv)
])
model.fit(X_train, y_train)
Predict and calculate accuracy
In [ ]:
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {round(acc * 100, 2)}")
2. Save XGBoost model to a file

We should save the created model object to a local path. The Github action will then take the model object from this path and upload it to Algorithmia.

In [ ]:
joblib.dump(model, "model.pkl", compress=True)
3. Test our serving (Algorithm) code

We will now test our algorithm code, ie. xgboost_automated.py script, by simply executing it with the %run macro. The script will begin its execution through our if __name__ == "__main__" line.

In [ ]:
%run xgboost_automated/src/xgboost_automated.py
Final checks before committing this notebook

Make sure xgboost_automated.py file is in good shape to be pushed to Algorithmia.

Make sure Algorithmia API Key is not committed and pushed in any file!

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from string import punctuation
from nltk.corpus import stopwords

from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib

data = pd.read_csv("./data/amazon_musical_reviews/Musical_instruments_reviews.csv")
data.head()

data["reviewText"].iloc[1]

import nltk
nltk.download('stopwords')

def threshold_ratings(data):
    def threshold_overall_rating(rating):
        return 0 if int(rating)<=3 else 1
    data["overall"] = data["overall"].apply(threshold_overall_rating)

def remove_stopwords_punctuation(data):
    data["review"] = data["reviewText"] + data["summary"]

    puncs = list(punctuation)
    stops = stopwords.words("english")

    def remove_stopwords_in_str(input_str):
        filtered = [char for char in str(input_str).split() if char not in stops]
        return ' '.join(filtered)

    def remove_punc_in_str(input_str):
        filtered = [char for char in input_str if char not in puncs]
        return ''.join(filtered)

    def remove_stopwords_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_stopwords_in_str(input_series[i]))
        return text_clean

    def remove_punc_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_punc_in_str(input_series[i]))
        return text_clean

    data["review"] = remove_stopwords_in_series(data["review"].str.lower())
    data["review"] = remove_punc_in_series(data["review"].str.lower())

def drop_unused_colums(data):
    data.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime', "reviewText", "summary"], axis=1, inplace=True)

def preprocess_reviews(data):
    remove_stopwords_punctuation(data)
    threshold_ratings(data)
    drop_unused_colums(data)

preprocess_reviews(data)
data.head()

rand_seed = 42
X = data["review"]
y = data["overall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)

params = {"max_depth": range(9,12), "min_child_weight": range(5,8)}
rand_search_cv = RandomizedSearchCV(XGBClassifier(), param_distributions=params, n_iter=1)

model  = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', rand_search_cv)
])
model.fit(X_train, y_train)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {round(acc * 100, 2)}")

joblib.dump(model, "model.pkl", compress=True)

%run xgboost_automated/src/xgboost_automated.py

from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib

data = pd.read_csv("./data/amazon_musical_reviews/Musical_instruments_reviews.csv")
data.head()

data["reviewText"].iloc[1]

import nltk
nltk.download('stopwords')

def threshold_ratings(data):
    def threshold_overall_rating(rating):
        return 0 if int(rating)<=3 else 1
    data["overall"] = data["overall"].apply(threshold_overall_rating)

def remove_stopwords_punctuation(data):
    data["review"] = data["reviewText"] + data["summary"]

    puncs = list(punctuation)
    stops = stopwords.words("english")

    def remove_stopwords_in_str(input_str):
        filtered = [char for char in str(input_str).split() if char not in stops]
        return ' '.join(filtered)

    def remove_punc_in_str(input_str):
        filtered = [char for char in input_str if char not in puncs]
        return ''.join(filtered)

    def remove_stopwords_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_stopwords_in_str(input_series[i]))
        return text_clean

    def remove_punc_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_punc_in_str(input_series[i]))
        return text_clean

    data["review"] = remove_stopwords_in_series(data["review"].str.lower())
    data["review"] = remove_punc_in_series(data["review"].str.lower())

def drop_unused_colums(data):
    data.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime', "reviewText", "summary"], axis=1, inplace=True)

def preprocess_reviews(data):
    remove_stopwords_punctuation(data)
    threshold_ratings(data)
    drop_unused_colums(data)

preprocess_reviews(data)
data.head()

rand_seed = 42
X = data["review"]
y = data["overall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)

params = {"max_depth": range(9,12), "min_child_weight": range(5,8)}
rand_search_cv = RandomizedSearchCV(XGBClassifier(), param_distributions=params, n_iter=1)

model  = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', rand_search_cv)
])
model.fit(X_train, y_train)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {round(acc * 100, 2)}")

joblib.dump(model, "model.pkl", compress=True)

%run xgboost_automated_github/src/xgboost_automated_github.py

---------------------------------------------------------------------------
DataApiError                              Traceback (most recent call last)
~/repos/algorithmia_ci/demo_autodeploy_algo_on_github/xgboost_automated_github.py in &lt;module&gt;
     58 
     59 config = load_model_config()
---&gt; 60 xgb_path, xgb_obj = load_model(config)
     61 assert_model_md5(xgb_path)
     62 

~/repos/algorithmia_ci/demo_autodeploy_algo_on_github/xgboost_automated_github.py in load_model(config)
     35     &quot;&quot;&quot;Loads the model object from the file at model_filepath key in config dict&quot;&quot;&quot;
     36     model_path = config[&quot;model_filepath&quot;]
---&gt; 37     model_file = client.file(model_path).getFile().name
     38     model_obj = joblib.load(model_file)
     39     return model_file, model_obj

~/opt/anaconda3/lib/python3.8/site-packages/Algorithmia/datafile.py in getFile(self)
     35         exists, error = self.existsWithError()
     36         if not exists:
---&gt; 37             raise DataApiError(&#39;unable to get file {} - {}&#39;.format(self.path, error))
     38         # Make HTTP get request
     39         response = self.client.getHelper(self.url)

DataApiError: unable to get file asli/xgboost_automated_github/model_1efcc5e311207268e450d6018debe7012f182e71.pkl - authorization required

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from string import punctuation
from nltk.corpus import stopwords

from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib

data = pd.read_csv("./data/amazon_musical_reviews/Musical_instruments_reviews.csv")
data.head()

data["reviewText"].iloc[1]

import nltk
nltk.download('stopwords')

def threshold_ratings(data):
    def threshold_overall_rating(rating):
        return 0 if int(rating)<=3 else 1
    data["overall"] = data["overall"].apply(threshold_overall_rating)

def remove_stopwords_punctuation(data):
    data["review"] = data["reviewText"] + data["summary"]

    puncs = list(punctuation)
    stops = stopwords.words("english")

    def remove_stopwords_in_str(input_str):
        filtered = [char for char in str(input_str).split() if char not in stops]
        return ' '.join(filtered)

    def remove_punc_in_str(input_str):
        filtered = [char for char in input_str if char not in puncs]
        return ''.join(filtered)

    def remove_stopwords_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_stopwords_in_str(input_series[i]))
        return text_clean

    def remove_punc_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_punc_in_str(input_series[i]))
        return text_clean

    data["review"] = remove_stopwords_in_series(data["review"].str.lower())
    data["review"] = remove_punc_in_series(data["review"].str.lower())

def drop_unused_colums(data):
    data.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime', "reviewText", "summary"], axis=1, inplace=True)

def preprocess_reviews(data):
    remove_stopwords_punctuation(data)
    threshold_ratings(data)
    drop_unused_colums(data)

preprocess_reviews(data)
data.head()

rand_seed = 42
X = data["review"]
y = data["overall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)

params = {"max_depth": range(9,12), "min_child_weight": range(5,8)}
rand_search_cv = RandomizedSearchCV(XGBClassifier(), param_distributions=params, n_iter=1)

model  = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', rand_search_cv)
])
model.fit(X_train, y_train)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {round(acc * 100, 2)}")

joblib.dump(model, "model.pkl", compress=True)

%run xgboost_automated/src/xgboost_automated.py
