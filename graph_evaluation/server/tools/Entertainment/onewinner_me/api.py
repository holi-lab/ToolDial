import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def registeremcomb(star2: int, num1: int, num2: int, num3: int, num4: int, num5: int, star1: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method will register a Euro Millions (EM) lottery combination. Registering a combination will prevent other players from choosing the same. 
		Euro Millions is the biggest lottery in the worlds. This pan European lottery game is run in 10 countries at the same time. Approximately 20 million people play every week in Europe.
		A Euro Millions combination is composed of 5 numbers and 2 stars."
    star2: 2nd number of the lottery combination you want to register. Between 1 and 11.
        num1: 1st number of the lottery combination you want to register. Between 1 and 50.
        num2: 2nd number of the lottery combination you want to register. Between 1 and 50.
        num3: 3rd number of the lottery combination you want to register. Between 1 and 50.
        num4: 4th number of the lottery combination you want to register. Between 1 and 50.
        num5: 5th number of the lottery combination you want to register. Between 1 and 50.
        star1: 1st star of the lottery combination you want to register. Between 1 and 11.
        
    """
    url = f"https://onewinnerme-onewinnerme.p.rapidapi.com/registeremcomb/{num1}/{num2}/{num3}/{num4}/{num5}/{star1}/{star2}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onewinnerme-onewinnerme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getemscore(num1: int, num2: int, num3: int, num4: int, num5: int, star1: int, star2: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method will return the score (earning expectation) of a Euro Millions (EM) lottery combination. 
		Euro Millions is the biggest lottery in the worlds. This pan European lottery game is run in 10 countries at the same time. Approximately 20 million people play every week in Europe.
		At Euro Millions combination is composed of 5 numbers and 2 stars."
    num1: 1st number of the lottery combination you want to check. Between 1 and 50.
        num2: 2nd number of the lottery combination you want to check. Between 1 and 50.
        num3: 3rd number of the lottery combination you want to check. Between 1 and 50.
        num4: 4th number of the lottery combination you want to check. Between 1 and 50.
        num5: 5th number of the lottery combination you want to check. Between 1 and 50.
        star1: 1st star of the lottery combination you want to check. Between 1 and 11.
        star2: 2nd star of the lottery combination you want to check. Between 1 and 1.
        
    """
    url = f"https://onewinnerme-onewinnerme.p.rapidapi.com/getemscore/{num1}/{num2}/{num3}/{num4}/{num5}/{star1}/{star2}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onewinnerme-onewinnerme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

