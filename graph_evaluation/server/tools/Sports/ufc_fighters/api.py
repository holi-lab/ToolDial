import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_fight_history(last_name: str, first_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of fights of the fighters
		Each fight includes:
		result, opponent, knockdowns, opponent knockdowns, strikes, opponent strikes, takedowns, opponent takedowns, submission attempts, opponent submission attemps, event, date, end method, end round, end time, whether it is a title fight."
    
    """
    url = f"https://ufc-fighters.p.rapidapi.com/fighters/history"
    querystring = {'last_name': last_name, 'first_name': first_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ufc-fighters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fighter_advanced_stats(first_name: str, last_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the advanced stats of a fighter:
		
		SLpM - Significant Strikes Landed per Minute
		Str. Acc. - Significant Striking Accuracy
		SApM - Significant Strikes Absorbed per Minute
		Str. Def. - Significant Strike Defence (the % of opponents strikes that did not land)
		TD Avg. - Average Takedowns Landed per 15 minutes
		TD Acc. - Takedown Accuracy
		TD Def. - Takedown Defense (the % of opponents TD attempts that did not land)
		Sub. Avg. - Average Submissions Attempted per 15 minutes"
    
    """
    url = f"https://ufc-fighters.p.rapidapi.com/fighters/advanced_stats"
    querystring = {'first_name': first_name, 'last_name': last_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ufc-fighters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_fighter(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random UFC fighter"
    
    """
    url = f"https://ufc-fighters.p.rapidapi.com/fighters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ufc-fighters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fighter_by_name(first_name: str, last_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fighter with query strings first_name and last_name"
    
    """
    url = f"https://ufc-fighters.p.rapidapi.com/fighters/search"
    querystring = {'first_name': first_name, 'last_name': last_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ufc-fighters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_fighters(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all UFC fighters"
    
    """
    url = f"https://ufc-fighters.p.rapidapi.com/fighters/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ufc-fighters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fighters_by_weight_class(weight_class: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all fighters in weight class
		Options:
		  strawweight, flyweight, bantamweight, featherweight, 
		  lightweight, welterweight, middleweight, lightheavyweight,
		  heavyweight"
    
    """
    url = f"https://ufc-fighters.p.rapidapi.com/fighters/by_weight_class/{weight_class}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ufc-fighters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_champions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current champions of all weight classes in the UFC"
    
    """
    url = f"https://ufc-fighters.p.rapidapi.com/fighters/champions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ufc-fighters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

