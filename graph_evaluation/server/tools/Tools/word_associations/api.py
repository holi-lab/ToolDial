import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def word_associations_get(lang: str, text: str, limit: int=50, indent: str='yes', pos: str='noun,adjective,verb,adverb', type: str='stimulus', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets associations with the given word or phrase."
    lang: Query language. Use language code for the language of the text: de - German; en - English; es - Spanish; fr - French; it - Italian; pt - Portuguese; ru - Russian;
        text: Word or phrase to find associations with.
Tip. You can use multiple parameters 'text' in a request (from 1 to 10 inclusive). This way you can get associations for several input words or phrases in one response.
Restriction: regardless of the size of the text association lookup is always performed by the first 10 words of the text.
        limit: Maximum number of results to return.
Allows to limit the number of results (associations) in response.
The value of this parameter is an integer number from 1 to 300 inclusive.
        indent: Indentation switch for pretty printing of JSON response.Allows to either turn on or off space indentation for a response.The following values are allowed: yes - turns indentation with spaces on; no - turn indentation with spaces off;
        pos: Parts of speech to return. Allows to limit results by specified parts of speech. The value of this parameter is a list of parts of speech separated by comma.The following parts of speech codes are supported: noun, adjective, verb, adverb.
        type: Type of result.Possible values:  'stimulus' - an input data (the text parameter) is considered as a response word. The service returns a list of stimuli words, which evoke a given response word; 'response' - an input data (the text parameter) is considered as a stimulus word. The service returns a list of response words, which come to mind for a given stimulus word.
        
    """
    url = f"https://wordassociations-word-associations-v1.p.rapidapi.com/json/search"
    querystring = {'lang': lang, 'text': text, }
    if limit:
        querystring['limit'] = limit
    if indent:
        querystring['indent'] = indent
    if pos:
        querystring['pos'] = pos
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordassociations-word-associations-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

