
import html
from diffusers.utils import is_ftfy_available
import regex as re
if is_ftfy_available():
    import ftfy

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text