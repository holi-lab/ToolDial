import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searching_through_posts(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can specify a keyword to search
		  - search=**keyword** returns posts that are talking about **keyword** or they just contain the **keyword** it self.
		- keyword accepts both capital letters and small letters."
    
    """
    url = f"https://programming-posts.p.rapidapi.com/"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "programming-posts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_posts_by_the_amount_you_need(amount: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can specify amount you want by:
		  - amount=**limit** returns **1000** posts.
		  - amount=**123** returns **123** posts.
		*parameter accepts only numbers and "limit" word, 
		if number passed the limit which is 1000 response will be 1000 posts.*"
    
    """
    url = f"https://programming-posts.p.rapidapi.com/"
    querystring = {}
    if amount:
        querystring['amount'] = amount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "programming-posts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

