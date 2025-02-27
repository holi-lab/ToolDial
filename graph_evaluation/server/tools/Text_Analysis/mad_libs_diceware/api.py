import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def madlibs_wordlist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get the wordlist used by madlibs diceware in JSON format**"
    
    """
    url = f"https://mad-libs-diceware.p.rapidapi.com/madlibs-wordlist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mad-libs-diceware.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eff_wordlist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get the wordlist used by plain diceware in JSON format**"
    
    """
    url = f"https://mad-libs-diceware.p.rapidapi.com/eff-wordlist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mad-libs-diceware.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def madlibs_diceware(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Generate password using madlibs diceware**
		
		Use query string parameter *nphrase* to change number of phrases in the password
		
		The generated phrase/s have the following structure: 
		
		```<adverb> <adjective> <noun>```
		
		Accepted values for *nphrases* parameter: 1 and 2"
    
    """
    url = f"https://mad-libs-diceware.p.rapidapi.com/madlibs-diceware"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mad-libs-diceware.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eff_diceware(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Generate password using plain diceware**
		
		Use query string parameter *nwords* to change number of words in the password
		
		Accepted values for *nwords* parameter: 4 to 7"
    
    """
    url = f"https://mad-libs-diceware.p.rapidapi.com/eff-diceware"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mad-libs-diceware.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

