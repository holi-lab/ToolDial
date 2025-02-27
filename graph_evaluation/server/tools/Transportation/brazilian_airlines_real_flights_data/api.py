import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_airline_data(company: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In this endpoint you get all data about real flights of airline. Including detailed information like flight plan routes and remarks"
    
    """
    url = f"https://brazilian-airlines-real-flights-data.p.rapidapi.com/flights"
    querystring = {'company': company, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brazilian-airlines-real-flights-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

