import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def n_chste_schulferien(limit: int=1, lang: str='de', state: str='NI', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Abfrage der nächsten Schulferien
		
		Mögliche Filter sind
		- Bundesland (Kürzel z.B. BY)
		- Anzahl (Limit)
		- Sprache (de, en)"
    limit: default = 1
        lang: default = de
        state: Kürzel Bundesland (BY, NI, NW, SL ...)
        
    """
    url = f"https://schulferien-und-feiertage.p.rapidapi.com/school-holidays/next"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if lang:
        querystring['lang'] = lang
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schulferien-und-feiertage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schulferien(school_year: str='2020_2021', year: int=2020, lang: str='de', state: str='BY', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Abfrage aller Schulferien-Termine mit diversen Filteroptionen
		
		Mögliche Filter sind 
		- Bundesland (Kürzel z.B. BY)
		- Jahr (z.B. 2020)
		- Schuljahr (z.B. 2020_2021)
		- Sprache (de, en)"
    lang: default = de
        state: Kürzel Bundesland (BY, NI, NW, SL ...)
        
    """
    url = f"https://schulferien-und-feiertage.p.rapidapi.com/school-holidays"
    querystring = {}
    if school_year:
        querystring['school_year'] = school_year
    if year:
        querystring['year'] = year
    if lang:
        querystring['lang'] = lang
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schulferien-und-feiertage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def derzeitige_schulferien(lang: str='de', state: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Abfrage der aktuellen/jetzigen Schulferien
		
		Mögliche Filter sind
		- Bundesland (Kürzel z.B. BY)
		- Sprache (de, en)"
    state: Kürzel Bundesland (BY, NI, NW, SL ...)
        
    """
    url = f"https://schulferien-und-feiertage.p.rapidapi.com/school-holidays/current"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schulferien-und-feiertage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

