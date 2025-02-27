import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bymarkup(targetmarkuppercentage: int, cost: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the raw price, suggested retail price, suggested low cost leader price, and the psychology price computed."
    targetmarkuppercentage: This is the targeted markup for sales prices.
        cost: This is the cost of the product which is used to calculate possible prices based on the required markup.
        
    """
    url = f"https://retail-pricing-optimizer.p.rapidapi.com/ByMarkUp"
    querystring = {'TargetMarkupPercentage': targetmarkuppercentage, 'Cost': cost, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "retail-pricing-optimizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bymargin(targetmarginpercentage: str, cost: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows the calculation of all possible prices by providing the cost and the targeted margin."
    targetmarginpercentage: This is the expected margin in percentage, it has to be positive.
        cost: This is the cost of the product, it has to be positive and can be 2 decimal points.
        
    """
    url = f"https://retail-pricing-optimizer.p.rapidapi.com/ByMargin"
    querystring = {'TargetMarginPercentage': targetmarginpercentage, 'Cost': cost, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "retail-pricing-optimizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def byminimumprice(minimumprice: str, cost: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates margins, markups, and suggested prices by providing both product cost and a target minimum sales price."
    minimumprice: This is the minimum price targeted by the service. All other prices will be based on the minimum price.
        cost: This is the product cost used to calculate margins and markups.
        
    """
    url = f"https://retail-pricing-optimizer.p.rapidapi.com/ByMinimumPrice"
    querystring = {'MinimumPrice': minimumprice, 'Cost': cost, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "retail-pricing-optimizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

