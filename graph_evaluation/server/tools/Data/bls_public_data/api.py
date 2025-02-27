import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_1_single_series(series_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this signature to retrieve data for a single time series for the past three years. Be sure to include the specific series ID at the end of the URL."
    series_id: he series ID(s) can include underscore (_), dash (-) and hash (#) but must not include lower case letters or special characters. All requests specifying years must include a four-digit start year and an end year in numeric format – YYYY (e.g. 2013, not 13).  See http://www.bls.gov/developers/api_signature.htm
        
    """
    url = f"https://community-bls-public-data.p.rapidapi.com/timeseries/data/{series_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-bls-public-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

