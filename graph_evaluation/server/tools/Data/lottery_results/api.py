import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_results_by_state(countrycode: str, statecode: str='ny', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can get the results from the latest draws for a given state.
		You need to provide the country code, and the state code is optional. 
		If you don't provide a state code you'll get a list of results from games that are played in multiple states.
		
		The state code should follow the ISO 3166-2, providing the 2nd part of the code as the state code.
		
		You can get a list of available state codes in the List states endpoint."
    
    """
    url = f"https://lottery-results.p.rapidapi.com/games-by-state/{countrycode}/{statecode}"
    querystring = {}
    if statecode:
        querystring['stateCode'] = statecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mega_millions_and_powerball(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoints returns the results from the two biggest games in the US. It exists just for compatibilty reasons."
    
    """
    url = f"https://lottery-results.p.rapidapi.com/lotteries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_states_by_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoints allows to get a list of states available for a country.
		 The country code should be given following ISO 3166-1 alpha-2 codes. 
		Currently we only support the US, so you should give the US as the country code to get the list of states."
    
    """
    url = f"https://lottery-results.p.rapidapi.com/list-states/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_games(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the results from all games by country code. Following the ISO 2 letter naming."
    
    """
    url = f"https://lottery-results.p.rapidapi.com/all-games/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

