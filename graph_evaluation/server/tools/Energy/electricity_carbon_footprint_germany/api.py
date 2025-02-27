import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def co2_emission(date: str='2022-08-20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The CO2 Emission endpoint will provide carbon emissions in grams per Kilo-Watt-Hour.  To latest and historical footprint information for electricity in Germany."
    date: Allows retrieving historical values (back to yr 2017) .

Accepted formats:

- YYYY-MM-DD
- MM/DD/YYYY
- Unix Timestamp (ms)
        
    """
    url = f"https://electricity-carbon-footprint-germany.p.rapidapi.com/gridde"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electricity-carbon-footprint-germany.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

