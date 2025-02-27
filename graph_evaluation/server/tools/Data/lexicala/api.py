import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searchentries(sample: int=None, source: str=None, language: str=None, analyzed: bool=None, subcategorization: str=None, polysemous: bool=None, pos: str=None, gender: str=None, monosemous: bool=None, morph: bool=None, text: str=None, number: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search for entries with filters in query"
    sample: Number of randomly-sampled results to return
        source: The resource to search within. The default value is 'global', i.e. the Global series
        language: The language code of the entry’s language. For an extensive list of language codes, see GET /languages
        analyzed: Whether to search using the language analyzer or to get exact matches only. The default value is 'false'
        subcategorization: The subcategorization (e.g. countable, transitive, …) to search for
        polysemous: Whether to only return polysemous entries
        pos: The part of speech (POS) to search for
        gender: The grammatical gender to search for
        monosemous: Whether to only return monosemous entries
        morph: Whether to search in headword inflections if exist. The default value is 'false'
        text: The headword text to search for
        number: The grammatical number to search for
        page: Page number of results to return (1-indexed). The default value is 1
        
    """
    url = f"https://lexicala1.p.rapidapi.com/search-entries"
    querystring = {}
    if sample:
        querystring['sample'] = sample
    if source:
        querystring['source'] = source
    if language:
        querystring['language'] = language
    if analyzed:
        querystring['analyzed'] = analyzed
    if subcategorization:
        querystring['subcategorization'] = subcategorization
    if polysemous:
        querystring['polysemous'] = polysemous
    if pos:
        querystring['pos'] = pos
    if gender:
        querystring['gender'] = gender
    if monosemous:
        querystring['monosemous'] = monosemous
    if morph:
        querystring['morph'] = morph
    if text:
        querystring['text'] = text
    if number:
        querystring['number'] = number
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lexicala1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def test(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "test that the API is running"
    
    """
    url = f"https://lexicala1.p.rapidapi.com/test"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lexicala1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "all supported languages and codes"
    
    """
    url = f"https://lexicala1.p.rapidapi.com/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lexicala1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def senses(sense_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get specific sense by its unique ID"
    sense_id: The sense ID of the sense
        
    """
    url = f"https://lexicala1.p.rapidapi.com/senses/{sense_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lexicala1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def entries(entry_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get specific entry by its unique ID"
    entry_id: The entry ID of the entry
        
    """
    url = f"https://lexicala1.p.rapidapi.com/entries/{entry_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lexicala1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(text: str=None, page: int=None, number: str=None, monosemous: bool=None, language: str=None, analyzed: bool=None, sample: int=None, pos: str=None, subcategorization: str=None, morph: bool=None, source: str=None, gender: str=None, polysemous: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search for entries with filters in query"
    text: The headword text to search for
        page: Page number of results to return (1-indexed). The default value is 1
        number: The grammatical number to search for
        monosemous: Whether to only return monosemous entries
        language: The language code of the entry’s language. For an extensive list of language codes, see GET /languages
        analyzed: Whether to search using the language analyzer or to get exact matches only. The default value is 'false'
        sample: Number of randomly-sampled results to return
        pos: The part of speech (POS) to search for
        subcategorization: The subcategorization (e.g. countable, transitive, …) to search for
        morph: Whether to search in headword inflections if exist. The default value is 'false'
        source: The resource to search within. The default value is 'global', i.e. the Global series
        gender: The grammatical gender to search for
        polysemous: Whether to only return polysemous entries
        
    """
    url = f"https://lexicala1.p.rapidapi.com/search"
    querystring = {}
    if text:
        querystring['text'] = text
    if page:
        querystring['page'] = page
    if number:
        querystring['number'] = number
    if monosemous:
        querystring['monosemous'] = monosemous
    if language:
        querystring['language'] = language
    if analyzed:
        querystring['analyzed'] = analyzed
    if sample:
        querystring['sample'] = sample
    if pos:
        querystring['pos'] = pos
    if subcategorization:
        querystring['subcategorization'] = subcategorization
    if morph:
        querystring['morph'] = morph
    if source:
        querystring['source'] = source
    if gender:
        querystring['gender'] = gender
    if polysemous:
        querystring['polysemous'] = polysemous
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lexicala1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

