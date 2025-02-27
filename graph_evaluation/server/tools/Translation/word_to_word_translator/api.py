import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def word_translation(limit: int, source: str, score: int, word: str, target: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`translate` is the core API of our service.
		
		Note that `translate` is not suited for sentences or longer texts. There are way better services for that (like Google translation API etc)'. Also note that: 
		
		> This is to be used as *a multilingual search engine or depending on your wishes, other natural language processing purposes and not for mere translation*.
		
		Although you can do so, because translation is the core of this, **it is very very approximate**.
		
		A `cat` might be translated to other contextual terms in other languages as: `["animal", "mammal", "domestic"]`. This approximate translation is in fact blazingly fast, and paradoxically more suited for search engines.
		
		Finally, You maybe guessed that `target` can be an array, indeed, and the form to provide multiple targets is simply adding multiple `target` parameters.
		
		Like the following:
		
		```
		 curl --get -G 'https://YourWord2WordTranslator@RapidAPI/api/translate?' \
		              -d word=science \
		              -d target=en \
		              -d target=fr \
		              -d target=bg \
		              -d source=en \
		              -d limit=10 \
		              -d score=0.5
		```"
    
    """
    url = f"https://word-to-word-translator1.p.rapidapi.com/api/translate"
    querystring = {'limit': limit, 'source': source, 'score': score, 'word': word, 'target': target, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-to-word-translator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def meta(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`meta` is simply to track of the current information about our back-end.
		
		The **Word to Word Translator** has been tested in every way possible, although You can expect changes as the API is in active improvement.  This is why you would talk with the `meta` API as your own safe guard.
		
		- You might for instance only request our APIs when all our services are up.
		- You might test again when when we increment our deployment version.
		- You might check again for the current supported languages."
    
    """
    url = f"https://word-to-word-translator1.p.rapidapi.com/api/meta"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-to-word-translator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def language_detection(phrase: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`detect_language` is a facilitator API to detect language of texts.
		
		Note that this API is very accurate and at the same time very fast as it relies on state of art machine learning techniques and implementations."
    
    """
    url = f"https://word-to-word-translator1.p.rapidapi.com/api/detect_language"
    querystring = {'phrase': phrase, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-to-word-translator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

