import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cryptocurrency_price_predictions(frequency: str, period: str, crypto_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The predictions route is a GET endpoint that allows users to retrieve future price predictions for a specified cryptocurrency. The user can specify the frequency of the prediction (**hour**, **day**, **month**, **year**), the period of prediction (between 1 and 3652), and the cryptocurrency ID. The endpoint returns a JSON object containing the predicted prices in the requested format. If the requested data is already cached, the response will be returned from the cache, saving on computation time."
    
    """
    url = f"https://cryptocurrency-price-prediction-api.p.rapidapi.com/predictions/{frequency}/{period}/{crypto_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-price-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cryptocurrencies_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /cryptocurrencies route is used to retrieve a list of available cryptocurrencies that can be used to generate price predictions. This endpoint returns a list of all available cryptocurrencies along with their corresponding IDs and names. Users can use this information to ensure that they provide the correct ID when making requests to the /predictions endpoint."
    
    """
    url = f"https://cryptocurrency-price-prediction-api.p.rapidapi.com/cryptocurrencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrency-price-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

