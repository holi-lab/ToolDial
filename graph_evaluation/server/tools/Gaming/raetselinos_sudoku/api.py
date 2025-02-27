import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_sudokus(x_api_key: str, sudokucount: int, difficulty: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generates Sudokus at once. difficulty can be easy, medium, hard, insane, inhuman. sudokuCount specifies the number of Sudokus you want. For each generated Sudoku you are charged one credit."
    
    """
    url = f"https://raetselinos-sudoku.p.rapidapi.com/generate/sudoku/{difficulty}/{sudokucount}"
    querystring = {'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "raetselinos-sudoku.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

