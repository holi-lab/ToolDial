import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translate_a_word_new(query: str, langpair: str, wortart: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Looks up a word in a bilingual dictionary and returns all translations ordered by frequency. Optionally restricts the result for a given wordclass."
    query: any word (case insensitive)
        langpair: One of the following combinations (in lower case): de-en, de-es, de-nl, de-pl, de-it, de-cs, en-de, es-de, nl-de, pl-de, it-de cs-de
        wortart: One of the following parts of speech (in upper case): ADJ, ADV, AUX, EIGENNAME, ITJ, PHRASE, PRON, PREP, PREP_ART, PTK, KONJ, MODAL, NOMEN, NUM, VERB
        
    """
    url = f"https://petapro-translate-v1.p.rapidapi.com/"
    querystring = {'query': query, 'langpair': langpair, }
    if wortart:
        querystring['wortart'] = wortart
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "petapro-translate-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

