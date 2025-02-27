import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def jokes(category: str='Insults', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for jokes by category or at random
		
		Available categories for the category query string parameter are:
		
		Animal
		Other / Misc
		Bar
		One Liners
		Puns
		Lawyer
		Sports
		Medical
		News / Politics
		Men / Women
		Gross
		Blond
		Yo Momma
		Redneck
		Religious
		At Work
		College
		Lightbulb
		Children
		Insults
		Knock-Knock
		Tech
		Yo Mama
		Blonde"
    
    """
    url = f"https://joke-generator-3000.p.rapidapi.com/jokes"
    querystring = {}
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joke-generator-3000.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

