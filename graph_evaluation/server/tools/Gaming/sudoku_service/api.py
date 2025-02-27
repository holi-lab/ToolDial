import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create(withsolution: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a new Sudoku puzzle. Currently, this endpoint does not offer control over the difficulty level of the puzzle returned. It will be be made randomly, and can be anywhere from extremely easy to extremely difficult."
    withsolution: This optional parameter can be set to true to to make the endpoint return the solution along with the puzzle.
        
    """
    url = f"https://sudoku-service.p.rapidapi.com/v1/sudoku"
    querystring = {}
    if withsolution:
        querystring['withSolution'] = withsolution
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sudoku-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

