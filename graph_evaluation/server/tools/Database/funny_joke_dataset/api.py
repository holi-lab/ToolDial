import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_random_jokes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A dataset of English jokes
		
		Additional fields:
		id -- page ID on id.
		category -- see available categories here.
		title -- title of the joke.
		
		category:
		Animal
		At Work
		Bar
		Blond
		Children
		College
		Gross
		Insults
		Knock-Knock
		Lawyer
		Lightbulb
		Medical
		Men / Women
		News / Politics
		One Liners
		Puns
		Redneck
		Religious
		Sports
		Tech
		Yo Momma
		Other / Misc"
    
    """
    url = f"https://funny-joke-dataset.p.rapidapi.com/users"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funny-joke-dataset.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

