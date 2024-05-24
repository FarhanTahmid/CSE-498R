import string
from nltk.corpus import stopwords
import unicodedata

def preprocess_text(text):
  """
  This function preprocesses text by:
    - Lowercasing all characters
    - Removing URLs
    - Removing punctuation and special characters
    - Removing stopwords
    - Keeping only English characters

  Args:
      text: The text to preprocess (string)

  Returns:
      The preprocessed text (string)
  """

  # Lowercase all characters
  text = text.lower()

  # Remove URLs using regular expressions (adapt if needed for specific URL formats)
  import re
  text = re.sub(r"http\S+", "", text)

  # Remove punctuation and special characters
  text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

  # Remove stopwords (download NLTK stopwords corpus first if not available)
  stop_words = stopwords.words('english')
  text = " ".join([word for word in text.split() if word not in stop_words])

  # Keep only English characters using unicode check
  text = ''.join(c for c in text if (unicodedata.category(c) == 'Ll' or c.isspace()))

  return text

# Example usage
text = "This is some text with a URL (http://www.example.com) and punctuation!@#$%^&*. It also includes stopwords and non-English characters like 中文."
preprocessed_text = preprocess_text(text)
print(preprocessed_text)