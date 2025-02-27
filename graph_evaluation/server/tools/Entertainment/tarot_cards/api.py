import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def return_tarot_cards_with_certain_amounts_of_minor_and_major(minor: int, major: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`?major=N&minor=M` -- this will return N major arcanas and M minor arcanas"
    
    """
    url = f"https://tarot-cards1.p.rapidapi.com/tarot/"
    querystring = {'minor': minor, 'major': major, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tarot-cards1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_minor_arcana(minor: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`?minor=N` -- this will return only Minor arcanas equal to N"
    
    """
    url = f"https://tarot-cards1.p.rapidapi.com/tarot/"
    querystring = {'minor': minor, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tarot-cards1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_major_arcana(major: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`?major=N` -- this will return only Major arcanas equal to N"
    
    """
    url = f"https://tarot-cards1.p.rapidapi.com/tarot/"
    querystring = {'major': major, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tarot-cards1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_tarot_card(amount: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`?amount=N` -- this will return a mix of Major and Minor arcanas equal to N"
    
    """
    url = f"https://tarot-cards1.p.rapidapi.com/tarot/"
    querystring = {'amount': amount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tarot-cards1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

