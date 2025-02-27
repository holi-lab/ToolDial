import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_crime_rates_by_zip(zip: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Just with the ZIP code to analyze, you will be receiving a JSON object with an Overall Crime Scoring, and also a breakdown of different crimes that are assessed on the zone. 
		
		Get information like:
		
		- "Overall Crime Grade".
		- "Violent Crime Grade".
		- "Property Crime Grade".
		- "Other Crime Grade".
		- "Violent Crime Rates".
		- "Property Crime Rates".
		- "Other Crime Rates".
		- "Crime Rates Nearby".
		- "Similar Population Crime Rates"."
    zip: ZIP Code to retrieve crime data from.
        
    """
    url = f"https://crime-data-by-zipcode-api.p.rapidapi.com/crime_data"
    querystring = {'zip': zip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crime-data-by-zipcode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

