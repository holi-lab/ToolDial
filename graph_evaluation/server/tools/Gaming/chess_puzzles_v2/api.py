import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def advanced(number_of_puzzles: int=1, themes: str='kingsideAttack,middlegame', theme_search_type: str='AND', number_of_moves: int=4, opening_variation: str='Kings_Gambit_Accepted_Abbazia_Defense', rating: str='1200', opening_family: str='Kings_Gambit_Accepted', max_deviation: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Advanced search options"
    number_of_puzzles: Number of puzzles you will get
        themes: Themes of the puzzles
        theme_search_type: AND: if puzzles need to have all themes;  OR: if puzzles can have either theme
        number_of_moves: Total number of moves of the puzzles
        opening_variation: Puzzle's opening variation
        rating: Rating of the puzzles
        opening_family: Puzzle's opening family
        max_deviation: Maximum possible rating deviation
        
    """
    url = f"https://chess-puzzles2.p.rapidapi.com/advanced"
    querystring = {}
    if number_of_puzzles:
        querystring['number_of_puzzles'] = number_of_puzzles
    if themes:
        querystring['themes'] = themes
    if theme_search_type:
        querystring['theme_search_type'] = theme_search_type
    if number_of_moves:
        querystring['number_of_moves'] = number_of_moves
    if opening_variation:
        querystring['opening_variation'] = opening_variation
    if rating:
        querystring['rating'] = rating
    if opening_family:
        querystring['opening_family'] = opening_family
    if max_deviation:
        querystring['max_deviation'] = max_deviation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chess-puzzles2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def range(number_of_puzzles: int=1, max_deviation: int=100, min: int=1200, max: int=1600, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random puzzles within a specified rating range"
    number_of_puzzles: Number of puzzles you will get
        max_deviation: Maximum possible rating deviation
        min: Minimum rating the puzzle can have 
        max: Maximum rating the puzzle can have 
        
    """
    url = f"https://chess-puzzles2.p.rapidapi.com/range"
    querystring = {}
    if number_of_puzzles:
        querystring['number_of_puzzles'] = number_of_puzzles
    if max_deviation:
        querystring['max_deviation'] = max_deviation
    if min:
        querystring['min'] = min
    if max:
        querystring['max'] = max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chess-puzzles2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def is_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get  a chess puzzle by its id"
    id: Unique id assigned to a certain puzzle
        
    """
    url = f"https://chess-puzzles2.p.rapidapi.com/id/{is_id}"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chess-puzzles2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random(number_of_puzzles: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random chess puzzles"
    number_of_puzzles: Number of puzzles you will get
        
    """
    url = f"https://chess-puzzles2.p.rapidapi.com/random"
    querystring = {}
    if number_of_puzzles:
        querystring['number_of_puzzles'] = number_of_puzzles
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chess-puzzles2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

