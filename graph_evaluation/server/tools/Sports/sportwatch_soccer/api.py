import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def all_calendar(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this request you get all of future match of this season."
    
    """
    url = f"https://sportwatch-soccer.p.rapidapi.com/getAllCalendar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportwatch-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def results_of_the_season_by_competition(comp: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Whit this endpoint you can get the results of one competition you want.
		
		The code for each ligue is :
		
		cl : Champions League
		pl : Premier League
		pe1 : La Liga
		sai : Serie A
		bund1 : Bundesligua
		fl1 : Ligue 1"
    
    """
    url = f"https://sportwatch-soccer.p.rapidapi.com/getResults/{comp}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportwatch-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_matchs_in_live(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint return all match in live of Champions League, Liga etc.."
    
    """
    url = f"https://sportwatch-soccer.p.rapidapi.com/getAllLive"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportwatch-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_results_of_season(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request give you the results of all match of the current season in Champions League, Premier League, Liga, Bundesligua, Serie A and Ligue 1."
    
    """
    url = f"https://sportwatch-soccer.p.rapidapi.com/getAllResults"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportwatch-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_by_competiton(comp: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint give you the list of matchs in live for selected competition
		
		The code for each ligue is :
		
		cl : Champions League
		pl : Premier League
		pe1 : La Liga
		sai : Serie A
		bund1 : Bundesligua
		fl1 : Ligue 1"
    
    """
    url = f"https://sportwatch-soccer.p.rapidapi.com/getLiveByComp/{comp}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportwatch-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calendar_by_competition(comp: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get the calendar of the selected competition
		
		The code for each ligue is :
		
		cl : Champions League
		pl : Premier League
		pe1 : La Liga
		sai : Serie A
		bund1 : Bundesligua
		fl1 : Ligue 1"
    
    """
    url = f"https://sportwatch-soccer.p.rapidapi.com/getCalendar/{comp}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportwatch-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_stats(matchid: str, comp: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is useful to get statistics about a match
		
		The matchId can be found in any request as list of Live Match or in Calendar
		
		The code for each ligue is :
		
		cl : Champions League
		pl : Premier League
		pe1 : La Liga
		sai : Serie A
		bund1 : Bundesligua
		fl1 : Ligue 1"
    
    """
    url = f"https://sportwatch-soccer.p.rapidapi.com/getLiveStats/{comp}/{matchid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportwatch-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

