import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_range_of_verses(chapterid: int, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Responds with a collection of *Ayahs/Verses* in a specific *Chapter/Surah* along with original Arabic text, translation, transliteration and verse ID in JSON"
    
    """
    url = f"https://al-quran1.p.rapidapi.com/{chapterid}/{range}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-quran1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_base_information(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Responds with JSON including some properties of the Quran. *(deprecated)*"
    
    """
    url = f"https://al-quran1.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-quran1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entire_surah_chapter(chapterid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Possible values: ***1-114*** 
		Responds with entire Surah/Chapter of the Koran including all verses in the Surah and some additional information."
    chapterid: Represents a unique Surah/Chapter in the Koran.
**Min Value: *1***
**Max Value: *114***
        
    """
    url = f"https://al-quran1.p.rapidapi.com/{chapterid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-quran1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_specific_ayah_verse(chapterid: int, verseid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Responds with a specific *Ayah/Verse* in a specific *Chapter/Surah* along with original Arabic text, translation, transliteration and verse ID in JSON"
    chapterid: Refers to a *Surah/Chapter* in the Koran
**Min Value: *1***
**Max Value: *114***
        verseid: A valid *Ayah/verse* number from a specific chapter from the Quran
        
    """
    url = f"https://al-quran1.p.rapidapi.com/{chapterid}/{verseid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-quran1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_a_word_in_quran(searchterm: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Looks for a given keyword (English) in ***translation_eng*** key in the entire Koran and responds with all the verses containing the given keyword and toal matches in JSON"
    searchterm: Any given keyword or phrase you like to search for in the Koran
        
    """
    url = f"https://al-quran1.p.rapidapi.com/corpus/{searchterm}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-quran1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

