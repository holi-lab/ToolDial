import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_definitions(mot: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get definition of a word"
    mot: Set the word you want to search
        
    """
    url = f"https://dicolink.p.rapidapi.com/mot/{mot}/definitions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dicolink.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_lexical_field(mot: str, limite: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Lexical Field for a word"
    
    """
    url = f"https://dicolink.p.rapidapi.com/mot/{mot}/champlexical"
    querystring = {}
    if limite:
        querystring['limite'] = limite
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dicolink.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_antonyms(mot: str, limite: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get antonyms of a specific word"
    
    """
    url = f"https://dicolink.p.rapidapi.com/mot/{mot}/antonymes"
    querystring = {}
    if limite:
        querystring['limite'] = limite
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dicolink.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_citations(mot: str, limite: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get citations of a word"
    mot: Set the word you want to search
        
    """
    url = f"https://dicolink.p.rapidapi.com/mot/{mot}/citations"
    querystring = {}
    if limite:
        querystring['limite'] = limite
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dicolink.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_expressions(mot: str, limite: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Expression with a specific word"
    
    """
    url = f"https://dicolink.p.rapidapi.com/mot/{mot}/expressions"
    querystring = {}
    if limite:
        querystring['limite'] = limite
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dicolink.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_scrabble_score(mot: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Scrabble score for a word"
    
    """
    url = f"https://dicolink.p.rapidapi.com/mot/{mot}/scorescrabble"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dicolink.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_synonyms(mot: str, limite: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get synonyms of a specific word"
    
    """
    url = f"https://dicolink.p.rapidapi.com/mot/{mot}/synonymes"
    querystring = {}
    if limite:
        querystring['limite'] = limite
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dicolink.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_word(verbeconjugue: bool=None, minlong: str='5', maxlong: str='-1', avecdef: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random word"
    
    """
    url = f"https://dicolink.p.rapidapi.com/mots/motauhasard"
    querystring = {}
    if verbeconjugue:
        querystring['verbeconjugue'] = verbeconjugue
    if minlong:
        querystring['minlong'] = minlong
    if maxlong:
        querystring['maxlong'] = maxlong
    if avecdef:
        querystring['avecdef'] = avecdef
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dicolink.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

