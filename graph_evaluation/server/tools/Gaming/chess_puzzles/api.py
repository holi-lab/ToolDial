import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def puzzles(themestype: str='ALL', rating: int=1500, count: int=25, themes: str='["middlegame","advantage"]', playermoves: int=4, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The main access point for the API"
    
    """
    url = f"https://chess-puzzles.p.rapidapi.com/"
    querystring = {}
    if themestype:
        querystring['themesType'] = themestype
    if rating:
        querystring['rating'] = rating
    if count:
        querystring['count'] = count
    if themes:
        querystring['themes'] = themes
    if playermoves:
        querystring['playerMoves'] = playermoves
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chess-puzzles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def puzzle_by_id(is_id: str='HxxIU', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a single puzzle by Id"
    
    """
    url = f"https://chess-puzzles.p.rapidapi.com/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chess-puzzles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

