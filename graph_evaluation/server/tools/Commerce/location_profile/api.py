import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_locate_ip_address(ip_address: str, pretty: str='True', fields: str='ip_address,continent_code,country_code,country,county,county_code,country_name,region_name,region_code,city,city_name,postcode,zipcode,latitude,longitude,time_zone,population_density,state,area_code,income,weather', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    pretty: Pretty print json
        fields: Choose the fields you want to include
        
    """
    url = f"https://locationprofile.p.rapidapi.com/api/locate/{ip_address}"
    querystring = {}
    if pretty:
        querystring['pretty'] = pretty
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locationprofile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

