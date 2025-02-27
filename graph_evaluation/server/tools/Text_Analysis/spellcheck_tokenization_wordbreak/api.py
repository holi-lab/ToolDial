import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_v1_wordbreak(str: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Very fast. No spellcheck. Break a long string of characters with no spaces - into words, separated by spaces."
    str: String of characters. With or without spaces. Punctuation is allowed, but is currently experimental.
        
    """
    url = f"https://spellcheck-tokenization-wordbreak.p.rapidapi.com/v1/wordbreak"
    querystring = {'str': str, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spellcheck-tokenization-wordbreak.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_v1_spellcheck(str: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Input a string of characters, with or without spaces. Receive a spell-checked phrase.
		
		This sends concurrent requests to both Bing/Google, then interprets which one (or neither) have the best spell check phrase. Returns the best string of the two, or the original string if both sources deemed not relevant.
		
		DOCUMENTATION at https://wordio.co/api"
    str: String of characters. With or without spaces. Punctuation is allowed, but is currently experimental.
        
    """
    url = f"https://spellcheck-tokenization-wordbreak.p.rapidapi.com/v1/spellcheck"
    querystring = {'str': str, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spellcheck-tokenization-wordbreak.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_v1_spellcheck_tokenize(str: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This makes a concurrent requests to both Bing and Google spellcheck APIs. This will add approximately 0.5 seconds to your wait time! If you'd rather not spell-check the phrase, and get your results much faster, use our /v1/wordbreak endpoint. Read more at https://wordio.co/api
		
		Input a string of characters, with or without spaces. Get back an array of words, with the part of speech, capitalization, and punctuation of each word. You actually get back multiple arrays of words. For example "unitedstatesofamerica" gets back ["United States"(n) "of" "America"(n)] and ["united"(adj), "states"(n), "of", "America"(n)].
		
		Punctuation inputs and outputs are currently experimental. If you'd like this endpoint to also include synonyms and derivations about each word, please contact the developer. https://wordio.co"
    str: String of characters. With or without spaces. Punctuation is allowed, but is currently experimental.
        
    """
    url = f"https://spellcheck-tokenization-wordbreak.p.rapidapi.com/v1/spellcheck-tokenize"
    querystring = {'str': str, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spellcheck-tokenization-wordbreak.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_v1_tokenize(str: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the answer much faster than /v1/spellcheck-wordbreak. However, it does not perform any spell-check. Read more at https://wordio.co/api
		
		Input a string of characters, with or without spaces. Get back an array of words, with the part of speech, capitalization, and punctuation of each word. You actually get back multiple arrays of words. For example "unitedstatesofamerica" gets back ["United States"(n) "of" "America"(n)] and ["united"(adj), "states"(n), "of", "America"(n)].
		
		Punctuation inputs and outputs are currently experimental. If you'd like this endpoint to also include synonyms and derivations about each word, please contact the developer. https://wordio.co"
    str: String of characters. With or without spaces. Punctuation is allowed, but is currently experimental.
        
    """
    url = f"https://spellcheck-tokenization-wordbreak.p.rapidapi.com/v1/tokenize"
    querystring = {'str': str, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spellcheck-tokenization-wordbreak.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

