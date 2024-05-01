import re
import bnlp

def clean_punctuations(text):
    # Defining a regular expression pattern to match Bangla punctuation marks
    bangla_punctuation = r'[।,?!.;:‘’“”\'\"‌‌—–-]'

    # Use re.sub() to replace punctuation marks with empty string
    cleaned_text = re.sub(bangla_punctuation, '', text)
    
    return cleaned_text

