import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def idlookup(country: str, source_id: str, source: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find out where to watch a movie or TV show by looking it up using common IDs (IMDb, tmdb, Wiki, etc).
		
		For example (L'affaire Dubuffet):
		IMDb, tt0331162"
    
    """
    url = f"https://movies-vista.p.rapidapi.com/idlookup/"
    querystring = {'country': country, 'source_id': source_id, 'source': source, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "movies-vista.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookup(country: str, term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a TV show or movie by name and check its availability through supported services for a specific country || Netflix, Amazon Prime Video, Amazon Instant Video, Apple TV +, Google Play, iTunes, YouTube Premium, Disney Plus, Hulu, Atom Tickets, CBS, DC Universe, HBO."
    
    """
    url = f"https://movies-vista.p.rapidapi.com/lookup/"
    querystring = {'country': country, 'term': term, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "movies-vista.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

