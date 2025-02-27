import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def your_kitchen_is_the_heart_of_your_home(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Your kitchen is the heart of your home, so why not make it as beautiful as it can be? With a Nivito kitchen faucet, you can have the perfect balance of form and function. This stunning kitchen faucet has a sleek, modern design that will complement any kitchen décor. But it's not just about looks – this faucet is also built to last, with a durable construction and a finish that resists tarnishing and corrosion. Plus, it comes with a convenient pull-down sprayer for all your cleaning and cooking needs. Whether you're washing dishes or prepping food, this faucet will make your life easier. And when you're not using it, the integrated magnetic docking system keeps"
    
    """
    url = f"https://nivito-singapore.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nivito-singapore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

