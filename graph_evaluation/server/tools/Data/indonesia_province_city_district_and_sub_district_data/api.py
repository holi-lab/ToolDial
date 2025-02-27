import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def village_kelurahan_or_desa_filtered_by_sub_district_bps_code(bps_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Village (Kelurahan or Desa) Filtered by  Sub-District BPS Code"
    
    """
    url = f"https://indonesia-province-city-district-and-sub-district-data.p.rapidapi.com/wilayah_kelurahan"
    querystring = {'bps_code': bps_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-province-city-district-and-sub-district-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sub_district_kecamatan_by_filtered_by_city_bps_code(bps_code: str='3101', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sub-District (Kecamatan) by Filtered by City BPS Code"
    
    """
    url = f"https://indonesia-province-city-district-and-sub-district-data.p.rapidapi.com/wilayah_kecamatan"
    querystring = {}
    if bps_code:
        querystring['bps_code'] = bps_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-province-city-district-and-sub-district-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_or_district_by_province_code(bps_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "City or District by Province Code"
    
    """
    url = f"https://indonesia-province-city-district-and-sub-district-data.p.rapidapi.com/wilayah_kabupaten"
    querystring = {'bps_code': bps_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-province-city-district-and-sub-district-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def province_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Province Data"
    
    """
    url = f"https://indonesia-province-city-district-and-sub-district-data.p.rapidapi.com/wilayah_propinsi"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-province-city-district-and-sub-district-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

