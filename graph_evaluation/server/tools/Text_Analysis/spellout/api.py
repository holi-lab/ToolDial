import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def languages(namefilter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List ISO 639 languages"
    namefilter: Filter as "contains" by language name
        
    """
    url = f"https://spellout.p.rapidapi.com/v1/languages"
    querystring = {}
    if namefilter:
        querystring['nameFilter'] = namefilter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spellout.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rulesets(lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of availible rule sets for given language"
    lang: 2 letter ICO 639 language code. Specifies language for which all availible rule sets will be provided. Use /v1/languages endpoint to list all supported languages.
        
    """
    url = f"https://spellout.p.rapidapi.com/v1/rulesets"
    querystring = {'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spellout.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spellout(data: int, lang: str, ruleset: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Spell out number in given language using provided rule set"
    data: Number to spell out
        lang: 2 letter ICO 639 language code. Specifies language in which number will be spelled out. Use /v1/languages endpoint to list all supported languages.
        ruleset: Rule Set name. Specifiles rule set accoring to which number will be spelled out. Use /v1/rulesets to list all supported rule sets for any given language.
        
    """
    url = f"https://spellout.p.rapidapi.com/v1/spellout"
    querystring = {'data': data, 'lang': lang, 'ruleset': ruleset, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spellout.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

