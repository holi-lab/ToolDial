import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def spellcheck(cmd: str='check_spelling', user_dictionary: str='Input_Dictionary_Name', text: str='Texxt with mispeled anyWord', slang: str='en_US', out_type: str='words', format: str='xml', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    slang: The parameter sets the default spell checking language for SCAYT. There are 16 languages available. Possible values are: 'en_US', 'en_GB', 'pt_BR', 'da_DK', 'nl_NL', 'en_CA', 'fi_FI', 'fr_FR', 'fr_CA', 'de_DE', 'el_GR', 'it_IT', 'nb_NO', 'pt_PT', 'es_ES', 'sv_SE'.
        out_type: Two possible values are acceptable: words and positions. Words - returns, misspelled words and suggestions, positions - returns position in text of misspellings and suggestions (default: words)
        format: Can be json and xml
        
    """
    url = f"https://webspellchecker-webspellcheckernet.p.rapidapi.com/ssrv.cgi"
    querystring = {}
    if cmd:
        querystring['cmd'] = cmd
    if user_dictionary:
        querystring['user_dictionary'] = user_dictionary
    if text:
        querystring['text'] = text
    if slang:
        querystring['slang'] = slang
    if out_type:
        querystring['out_type'] = out_type
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webspellchecker-webspellcheckernet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deletedictionary(cmd: str, action: str, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delete your User Dictionary"
    
    """
    url = f"https://webspellchecker-webspellcheckernet.p.rapidapi.com/ssrv.cgi"
    querystring = {'cmd': cmd, 'action': action, 'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webspellchecker-webspellcheckernet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deletewordfromuserdictionary(cmd: str, name: str, word: str, action: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can easily remove word from the User Dictionary"
    
    """
    url = f"https://webspellchecker-webspellcheckernet.p.rapidapi.com/ssrv.cgi"
    querystring = {'cmd': cmd, 'name': name, 'word': word, 'action': action, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webspellchecker-webspellcheckernet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def renameuserdictionary(cmd: str, name: str, new_name: str, action: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rename your User Dictionary"
    name: Input name of your existing User Dictionary
        new_name: Input new name for your User Dictionary
        
    """
    url = f"https://webspellchecker-webspellcheckernet.p.rapidapi.com/ssrv.cgi"
    querystring = {'cmd': cmd, 'name': name, 'new_name': new_name, 'action': action, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webspellchecker-webspellcheckernet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addwordtotheuserdictionary(cmd: str, action: str, name: str, word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request will add new word to the User Dictionary"
    action: Add word to the User Dictionary
        name: Specify the name of the User Dictionary where the word should be added
        word: Input word which should be added to the dictionary
        
    """
    url = f"https://webspellchecker-webspellcheckernet.p.rapidapi.com/ssrv.cgi"
    querystring = {'cmd': cmd, 'action': action, 'name': name, 'word': word, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webspellchecker-webspellcheckernet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def createuserdictionary(cmd: str, action: str, name: str, wordlist: str='Texxt,mispeled', format: str='xml', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API allows you to create the User Dictionary, so you are able to add any misspelling into your personal dictionary."
    action: Parameter 'action' is responsible for actions which you can do with your User Dictionary. Available values are: create, rename, delete, check, getdict.
        name: Give your User Dictionary a name
        wordlist: Coma-separated values that will be added to a created dictionary
        format: The following values are supported: xml and json
        
    """
    url = f"https://webspellchecker-webspellcheckernet.p.rapidapi.com/ssrv.cgi"
    querystring = {'cmd': cmd, 'action': action, 'name': name, }
    if wordlist:
        querystring['wordlist'] = wordlist
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webspellchecker-webspellcheckernet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def grammarcheck(cmd: str, text: str, slang: str, format: str='xml', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API allows you to check your grammar. This feature is available for American English and British English."
    text: Text to be checked
        slang: This parameter is responsible for a grammar checking language. Only two possible languages are supported: American English (en_US) and Great Britain English (en_GB).
        format: The following values are supported: xml and json
        
    """
    url = f"https://webspellchecker-webspellcheckernet.p.rapidapi.com/ssrv.cgi"
    querystring = {'cmd': cmd, 'text': text, 'slang': slang, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webspellchecker-webspellcheckernet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

