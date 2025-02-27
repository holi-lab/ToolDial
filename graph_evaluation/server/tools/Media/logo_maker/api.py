import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate(keyword: str, companyname: str, style: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It allows you to randomly generate 32 logos according to your directives, enter the name of the company or company, a keyword that indicates the area in which it will be present, for example: Sport and a type of style already specified in the values.
		There are 3 types of styles, 0: Logo with icon & text - 1: Logo with text only - 2: Mixed."
    keyword: Your sector, example: Sport
        companyname: You company name or your name, example: Tommy Gamer
        style: Your style, types:
0 -> Image & Text.
1 -> Only text.
2 -> Mixed.
        
    """
    url = f"https://logo-maker.p.rapidapi.com/Generate/{companyname}/{keyword}/{style}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "logo-maker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

