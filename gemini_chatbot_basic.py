import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


GOOGLE_API_KEY='AIzaSyDz4LwarLnZ9NAHmMrbkkT8JMeSatQoMq4'
genai.configure(api_key=GOOGLE_API_KEY)

for m in genai.list_models():
 ## print("*************")
  print(m.name)
  
    
