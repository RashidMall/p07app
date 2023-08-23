import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

EMOJIS = {
    ">:-(": "Angry",
    ">:)": "Smug",
    ">:D": "Evil Grin",
    ">;)": "Mischievous",
    "<:-)": "Party",
    ":')": "Tears of Joy",
    ":|]": "Robotic",
    ":-|]": "I'm a robot",
    "=^.^=": "Cat",
    ":^)": "Raised Eyebrow",
    ":-E": "Bucktooth",
    ":O)": "Clown",
    "<:3": "Kitty",
    "<3": "Love",
    ":D": "Grin",
    ":-D": "Grin",
    ":O": "Surprised",
    ":-O": "Surprised",
    ":)": "Smile",
    ":-)": "Smile",
    ":]": "Happy",
    ":-]": "Happy",
    ":P": "Tongue",
    ":-P": "Tongue",
    ":|(": "Displeased",
    ":(": "Sad",
    ":-(": "Sad",
    ":|": "Neutral",
    ":-|": "Neutral",
    ":/": "Confused",
    ":-/": "Confused",
    ";-)": "Wink",
    ";)": "Wink",
    ":*": "Kiss",
    ":-*": "Kiss",
    ":$": "Embarrassed",
    ":-$": "Embarrassed",
    ":Z": "Sleepy",
    ":B": "Cool",
    "B-)": "Cool"
}

def clean_text(text):
    """
    Clean the 'text'.

    """
    # Replace emojis
    clean_text = replace_emojis_with_descriptions(text, EMOJIS)    
    
    # Remove @USERNAME
    clean_text = remove_twitter_usernames(clean_text)
    
    # Convert text to lowercase
    clean_text = clean_text.lower()
    
    # Replace 3 or more consecutive letters by 2 letters
    clean_text = replace_consecutive_letters(clean_text)
    
    # Expand the contraction
    clean_text = expand_contractions(clean_text)

    # Remove URLs
    clean_text = remove_URL(clean_text)

    # Remove HTML tags
    clean_text = remove_html(clean_text)

    # Remove non-ASCII characters
    clean_text = remove_non_ascii(clean_text)

    # Remove special characters
    clean_text = remove_special_characters(clean_text)

    # Remove punctuation
    clean_text = remove_punct(clean_text)
    
    # Remove numbers
    clean_text = remove_numbers(clean_text)
    
    # Remove short words with length less than 2
    clean_text = remove_short_words(clean_text)

    return clean_text

def expand_contractions(text):
    """
    Expand contractions in a given text.

    Parameters:
    -----------
        - text (str): Input text containing contractions.

    Returns:
    --------
        - str: Text with contractions expanded.

    """
    # Dictionary of common English contractions and their expanded forms
    contractions_dict = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "I would",
        "i'd've": "I would have",
        "i'll": "I will",
        "i'll've": "I will have",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    # Regular expression pattern to find contractions
    pattern = re.compile(r"\b(" + "|".join(contractions_dict.keys()) + r")\b")

    # Function to replace contraction with expanded form
    def replace_contractions(match):
        contraction = match.group(0)
        expanded_form = contractions_dict[contraction]
        return expanded_form

    # Expand contractions using regex substitution
    expanded_text = re.sub(pattern, replace_contractions, text)

    return expanded_text

def replace_emojis_with_descriptions(text, emoji_dict):
    """
    Replaces emojis in the given text with their corresponding descriptions from the emoji dictionary.
    
    Parameters:
    -----------
        text (str): The text containing emojis.
        emoji_dict (dict): A dictionary mapping emojis to their descriptions.
    
    Returns:
    --------
        str: The text with emojis replaced by their descriptions.
    """
    for emoji, description in emoji_dict.items():
        text = text.replace(emoji, description)
    
    return text

def remove_twitter_usernames(text):
    """
    Removes Twitter usernames (starting with @) from the given text.
    
    Parameters:
    -----------
        - text (str): The text that may contain Twitter usernames.
    
    Returns:
    --------
        - str: The text with Twitter usernames removed.
    """
    # Regular expression pattern to match Twitter usernames
    username_pattern = re.compile(r'@(\w+)')
    
    # Remove Twitter usernames from the text using the pattern
    text_without_usernames = username_pattern.sub('', text)
    
    return text_without_usernames

def replace_consecutive_letters(text):
    """
    Replaces three or more consecutive letters with two letters in the given text.
    
    Parameters:
    -----------
        - text (str): The text that may contain consecutive letters.
    
    Returns:
    --------
        - str: The text with consecutive letters replaced.
    """
    # Regular expression pattern to match three or more consecutive letters
    consecutive_letters_pattern = re.compile(r'(\w)(\1{2,})', flags=re.IGNORECASE)
    
    # Replace consecutive letters with two letters
    text_with_replacement = consecutive_letters_pattern.sub(r'\1\1', text)
    
    return text_with_replacement

def remove_short_words(text):
    """
    Removes short words with a length less than 2 in the given text.
    
    Parameters:
    -----------
        - text (str): The text that may contain short words.
    
    Returns:
    --------
        - str: The text with short words removed.
    """
    # Regular expression pattern to match short words
    short_words_pattern = re.compile(r'\b\w{1}\b')
    
    # Remove short words from the text using the pattern
    text_without_short_words = short_words_pattern.sub('', text)
    
    return text_without_short_words

def remove_URL(text):
    """
        Remove URLs from a sample string
    """
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def remove_html(text):
    """
        Remove the html in sample text
    """
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)

def remove_non_ascii(text):
    """
        Remove non-ASCII characters 
    """
    return re.sub(r'[^\x00-\x7f]',r'', text)

def remove_special_characters(text):
    """
        Remove special special characters, including symbols, emojis, and other graphic characters
    """
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    """
        Remove the punctuation
    """
    #print(text)
    return re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)
    #return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text):
    """
        Remove numbers
    """
    return "".join(i for i in text if not i.isdigit())

def preprocess_text(text):
    """
    Preprocesses text data by performing cleaning, tokenization, stopword removal, and lemmatization.
    """
    # Clean the text
    cleaned_text = clean_text(text)

    # Tokenize the text
    tokenized_text = word_tokenize(cleaned_text)

    # Remove stopwords
    stopwords_removed = remove_stopwords(tokenized_text)

    # Lemmatize the text
    lemmatized_text = lemmatize_text(stopwords_removed)

    return lemmatized_text

def remove_stopwords(tokenized_text):
    """
    Remove stopwords from a tokenized text.

    Parameters:
    -----------
        - tokenized_text (pd.Series or list): Tokenized text data.

    Returns:
    --------
        - pd.Series or list: Tokenized text with stopwords removed.

    """
    stopwords = nltk_stopwords.words('english')
    stopwords_removed = [word for word in tokenized_text if word not in stopwords]
    return stopwords_removed

def lemmatize_text(text):
	"""
	Process the text by lemmatizing the words and removing stop words.

	"""
	# Lemmatize the words
	lemmatized_text = lemmatize_word(text)

	# Remove stop words
	lemmatized_text = remove_stopwords(lemmatized_text)

	# Join the lemmatized words back into sentences
	lemmatized_text = ' '.join(lemmatized_text)

	return lemmatized_text


def lemmatize_word(text):
	"""
	Lemmatize the tokenized words.

	Parameters:
	-----------
		- text (list): Tokenized words with / without POS tags.

	Returns:
	--------
		- list: Lemmatized words.

	"""
	lemmatizer = WordNetLemmatizer()
	lemmatized_words = []

	if isinstance(text, list):  # If the input is a simple list
		for word in text:
			lemma = lemmatizer.lemmatize(word)
			lemmatized_words.append(lemma)
	elif isinstance(text, tuple):  # If the input is a list of tuples (word, tag)
		for word, tag in text:
			if tag:
				lemma = lemmatizer.lemmatize(word, tag)
			else:
				lemma = lemmatizer.lemmatize(word)
			lemmatized_words.append(lemma)

	return lemmatized_words

# Load the trained model and vectorizer using Pickle
with open('lr_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf.pickle', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		tweet = request.form['tweet']
		
		preprocessed_tweet = preprocess_text(tweet)

		# Vectorize the input tweet
		tweet_vector = vectorizer.transform([preprocessed_tweet])

		# Predict sentiment using the loaded model
		sentiment = model.predict(tweet_vector)[0]
		
		if sentiment > 0.5:
			sentiment = "Positive"
		else:
			sentiment = "Bad Buzz !!!"
        
		return render_template('index.html', sentiment=sentiment, tweet=tweet)
    
	return render_template('index.html', sentiment=None, tweet=None)

if __name__ == "__main__":
    app.run(debug=True)