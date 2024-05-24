import re
import bnlp
import emoji
from bnlp import NLTKTokenizer
from bnlp import BengaliCorpus as corpus
# from bnlp.corpus.util import remove_stopwords

def clean_punctuations(text):
    
    # there might be some cases where might not be spaces after the full stop punctuation.
    
    # Defining a regular expression pattern to match "।<word>" without a space
    pattern = re.compile(r'\।(\w+)(?!\s|$)')
    # Replacing occurrences of "।<word>" without a space with "| <word>"
    modified_sentence = re.sub(pattern, r'। \1', text)
    
    # Defining a regular expression pattern to match Bangla punctuation marks
    bangla_punctuation = r'[।,?!.;:‘’“”\'\"‌‌—–-]'

    # Use re.sub() to replace punctuation marks with empty string
    cleaned_text = re.sub(bangla_punctuation, '', modified_sentence)
    
    return cleaned_text


def clean_emoji(text):
        # Defining a regular expression pattern to match emoji in Bangla Text
        
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
        # Remove emojis
        cleaned_text = emoji_pattern.sub(r'', text)
        return cleaned_text


def clean_digits(text):
    # Defining a regular expression pattern to match digits
    digit_pattern = r'\d'
    # replacing digits with empty string
    cleaned_text = re.sub(digit_pattern, '', text)
    return cleaned_text

def clean_url_and_email(text):
    # Defining a regular expression pattern to match URLs, URIs, and email addresses
    url_pattern = r'https?://\S+|www\.\S+|[\w\.-]+@[\w\.-]+'
    # Removing URLs, URIs, and email addresses
    cleaned_text = re.sub(url_pattern, '', text)
    return cleaned_text


def word_tokenize_texts(text):
    '''Returns tokenized words'''
    
    tokenizer=NLTKTokenizer()    
    tokens=tokenizer.word_tokenize(text)
    
    return tokens

def remove_stopwords_from_tokens(tokens):
    bangla_stopwords=corpus.stopwords
    filtered_tokenized_strings=[word for word in tokens if word not in bangla_stopwords]
    return filtered_tokenized_strings
