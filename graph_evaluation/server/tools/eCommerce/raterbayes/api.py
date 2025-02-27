import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def compare_ratings_for_two_amazon_products_using_urls(url2: str, url1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accepts two Amazon URLs, scrapes ratings data, and computes posterior probabilities for which product is better-rated on average"
    
    """
    url = f"https://raterbayes.p.rapidapi.com/test/amznURLs"
    querystring = {'URL2': url2, 'URL1': url1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "raterbayes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compare_ratings_directly(a3: int, a2: int, a5: int, a4: int, b5: int, b2: int, b3: int, b4: int, b1: int, a1: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Compares data for two ratings A and B, and returns a number describing the estimated posterior probability that the item associated with rating A is truly better than the one associated with rating B."
    
    """
    url = f"https://raterbayes.p.rapidapi.com/test/ratings"
    querystring = {'A3': a3, 'A2': a2, 'A5': a5, 'A4': a4, 'B5': b5, 'B2': b2, 'B3': b3, 'B4': b4, 'B1': b1, 'A1': a1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "raterbayes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

