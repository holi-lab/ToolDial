import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def data_country_all(package: str, typeodds: str, langodds: str, sportid: str, typeparams: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Data from all countries for all sports"
    package: Secret key
        typeodds: line or live
        langodds: en, ru, fr ....
        sportid: 0 - it is all countries for all sports
        typeparams: PArams: countries
        
    """
    url = f"https://sports-odds-betapi.p.rapidapi.com/v1/{typeparams}/{sportid}/{typeodds}/{langodds}"
    querystring = {'Package': package, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-odds-betapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def data_events(package: str, list_type: str, sport_id: int, tournament_id: int, page_length: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Data Events api"
    list_type: Way of formation of a conclusion. There may be two kinds: sub - events will be as sub-items of tournaments. list - all items are displayed, simply by a common list.
        sport_id: sport ID, if ID = 0, all matches for all sports will be returning
        tournament_id: tournament ID, if ID = 0, all matches for all tournament will be returning
        page_length: Number of events in answer.It must be a multiple of five. If it 's not like that. That, the system itself will occlude it to the greater side of the multiplicity.
        
    """
    url = f"https://sports-odds-betapi.p.rapidapi.com/v1/events/{sport_id}/{tournament_id}/{list_type}/{page_length}/line/ru"
    querystring = {'Package': package, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-odds-betapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def data_country(package: str, typeodds: str, sportid: int, langodds: str, typeparams: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Data Country params"
    typeodds: line, live
        sportid: number sport [1 or 2 or 3]. All data = 0
        langodds: en, ru, fr ....
        typeparams: sports, countries, tournaments
        
    """
    url = f"https://sports-odds-betapi.p.rapidapi.com/v1/{typeparams}/{sportid}/{typeodds}/{langodds}"
    querystring = {'Package': package, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-odds-betapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def data_sport(package: str, langodds: str, typeodds: str, typeparams: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Data Sport params"
    langodds: en, ru
        typeodds: line, live
        typeparams: sports, countries, tournaments
        
    """
    url = f"https://sports-odds-betapi.p.rapidapi.com/v1/{typeparams}/{typeodds}/{langodds}"
    querystring = {'Package': package, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-odds-betapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def data_tournaments(package: str, langodds: str, sportid: int, typeodds: str, typeparams: str, countryid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Data Tournaments"
    langodds: en, ru
        sportid: number sport [1 or 2 or 3]. All data = 0
        typeodds: line, live
        typeparams: sports, countries, tournaments
        countryid: number Country[1 or 2 or 3]. All data = 0
        
    """
    url = f"https://sports-odds-betapi.p.rapidapi.com/v1/{typeparams}/{sportid}/{countryid}/{typeodds}/{langodds}"
    querystring = {'Package': package, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-odds-betapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def data_tournaments_all(package: str, typeparams: str, sportid: str, langodds: str, countryid: int, typeodds: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Data Tournaments All Sports & All Countries"
    typeparams: sports, countries, tournaments
        sportid: All data = 0
        langodds: en, ru
        countryid: All data = 0
        typeodds: line, live
        
    """
    url = f"https://sports-odds-betapi.p.rapidapi.com/v1/{typeparams}/{sportid}/{countryid}/{typeodds}/{langodds}"
    querystring = {'Package': package, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-odds-betapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def concrete_match(package: str, dataapi: str, datatype: str, game_id: int, datalang: str, list_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "data of  concrete match"
    package: rewrewr
        dataapi: The "event" parameter indicates that we return a match with a specific id
        datatype: line or live
        game_id: enter the match id
        datalang: ru, en
        list_type: Way of formation of a conclusion. There may be two kinds: sub - events will be grouped into groups. list - all events are displayed, just a general list.
        
    """
    url = f"https://sports-odds-betapi.p.rapidapi.com/v1/{dataapi}/{game_id}/{list_type}/{datatype}/{datalang}"
    querystring = {'Package': package, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-odds-betapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

