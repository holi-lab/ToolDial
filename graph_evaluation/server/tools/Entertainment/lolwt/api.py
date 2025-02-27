import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getlistof(language: str, argument: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain list of type specified"
    language: en,fr,de,es or it
        argument: <type>=<mode> (possible type : item, rune, mastery, profileicon, champion, summoner, spell) (Possible mode : 0 default, 1 inverse key & value)
        
    """
    url = f"https://xavblog-lolwt.p.rapidapi.com/getListOf/{language}/{argument}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xavblog-lolwt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchampion(language: str, argument: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    language: en, fr, de, es or it
        argument: <id> Champion's id
        
    """
    url = f"https://xavblog-lolwt.p.rapidapi.com/getChampion/{language}/{argument}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xavblog-lolwt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmastery(language: str, argument: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    language: en, fr, de, es or it
        argument: <id> (Mastery's id)
        
    """
    url = f"https://xavblog-lolwt.p.rapidapi.com/getMastery/{language}/{argument}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xavblog-lolwt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getitem(language: str, argument: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To get all data about an Item"
    language: en, fr, de, es or it
        argument: <id> (Item's id)
        
    """
    url = f"https://xavblog-lolwt.p.rapidapi.com/getItem/{language}/{argument}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xavblog-lolwt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchampionspell(language: str, argument: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    language: en, fr, de, es or it
        argument: <id> Champion Spell's id
        
    """
    url = f"https://xavblog-lolwt.p.rapidapi.com/getChampionSpell/{language}/{argument}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xavblog-lolwt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlanguage(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the language Json of League of Legends in language specified"
    language: en,fr,de,es or it
        
    """
    url = f"https://xavblog-lolwt.p.rapidapi.com/getLanguage/{language}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xavblog-lolwt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def version(langue: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To obtain version of each type of Json data"
    langue: Language : en, fr, de, es or it
        
    """
    url = f"https://xavblog-lolwt.p.rapidapi.com/version/{langue}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xavblog-lolwt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrune(language: str, argument: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    language: en, fr, de, es or it
        argument: <id> (Rune's id)
        
    """
    url = f"https://xavblog-lolwt.p.rapidapi.com/getRune/{language}/{argument}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xavblog-lolwt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsummonerspell(language: str, argument: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    language: en, fr, de, es or it
        argument: <id> (Summoner's spell id)
        
    """
    url = f"https://xavblog-lolwt.p.rapidapi.com/getSummonerSpell/{language}/{argument}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xavblog-lolwt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

