import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_covid19(type: str=None, country: str=None, date: str='2022-01-01', county: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Covid19 API endpoint. Either **date** or **country** must be set."
    type: type of data to retrieve. Must be either cases or deaths. If not set, cases will be used by default.
        country: country name (case insensitive).
        date: date to retrieve single-day snapshot. Must be in the form of YYYY-MM-DD (e.g. 2022-01-01)
        county: county name for US states (case insensitive). For United States data only. Must be used in conjunction with country (United States) and region (e.g. California).
        region: administrative region (also known as state or province in many countries) name (case insensitive). Must be used in conjunction with country. If not set, countries with data broken down by administrative regions will return separate data for each region.
        
    """
    url = f"https://covid-19-by-api-ninjas.p.rapidapi.com/v1/covid19"
    querystring = {}
    if type:
        querystring['type'] = type
    if country:
        querystring['country'] = country
    if date:
        querystring['date'] = date
    if county:
        querystring['county'] = county
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

