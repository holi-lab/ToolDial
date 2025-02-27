import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calculate_combo_odds_uses_2_calls(profit: str, drawodd: str, totalgoalsover25odd: str, matchwinneroddonfavorite: str, favouritetypeonmatch: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate the basic and combo odds.
		
		Combo odds - odds in which you have to satisfy two conditions of Standard Odds in order to win"
    profit: A decimal number that represents the percent of profit you want to have on all of your odds.
        drawodd: The Draw result odd.
        totalgoalsover25odd: The Goals Over 2.5 Odd.
        matchwinneroddonfavorite: The match winner odd for the team that is the favorite.
For example, if the home team is the favorite then you would take the Match Winner 1 odd.
If the away team is the favorite then you would take the Match Winner 2 odd.
        favouritetypeonmatch: This parameter can only have two values:

1 - home team is favorite
2 - away team is favorite
        
    """
    url = f"https://sharp-odds-maker-api.p.rapidapi.com/calculate/{favouritetypeonmatch}/{matchwinneroddonfavorite}/{drawodd}/{totalgoalsover25odd}/{profit}/2"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sharp-odds-maker-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calculate_combo_odds_with_or_uses_3_calls(totalgoalsover25odd: str, favouritetypeonmatch: str, profit: str, matchwinneroddonfavorite: str, drawodd: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate the Basic, Combo and Combo Odds with Or.
		
		Combo Odds with Or - odds in which you have to satisfy only one of two conditions of Standard Odds to win"
    totalgoalsover25odd: The Goals Over 2.5 Odd.
        favouritetypeonmatch: This parameter can only have two values:

1 - home team is favorite
2 - away team is favorite
        profit: A decimal number that represents the percent of profit you want to have on all of your odds.
        matchwinneroddonfavorite: The match winner odd for the team that is the favorite.
For example, if the home team is the favorite then you would take the Match Winner 1 odd.
If the away team is the favorite then you would take the Match Winner 2 odd.
        drawodd: The Draw result odd.
        
    """
    url = f"https://sharp-odds-maker-api.p.rapidapi.com/calculate/{favouritetypeonmatch}/{matchwinneroddonfavorite}/{drawodd}/{totalgoalsover25odd}/{profit}/3"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sharp-odds-maker-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calculate_basic_odds_uses_1_call(drawodd: str, favouritetypeonmatch: str, profit: str, totalgoalsover25odd: str, matchwinneroddonfavorite: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate the basic odds."
    drawodd: The Draw result odd.
        favouritetypeonmatch: This parameter can only have two values:

1 - home team is favorite
2 - away team is favorite
        profit: A decimal number that represents the percent of profit you want to have on all of your odds.
        totalgoalsover25odd: The Goals Over 2.5 Odd.
        matchwinneroddonfavorite: The match winner odd for the team that is the favorite.
For example, if the home team is the favorite then you would take the Match Winner 1 odd.
If the away team is the favorite then you would take the Match Winner 2 odd.
        
    """
    url = f"https://sharp-odds-maker-api.p.rapidapi.com/calculate/{favouritetypeonmatch}/{matchwinneroddonfavorite}/{drawodd}/{totalgoalsover25odd}/{profit}/1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sharp-odds-maker-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

