import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_company_patents(x_hojininfo_api_token: str, accept: str, corporateid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows you to determine which patents have been filed by a Japanese company via their corporate ID"
    
    """
    url = f"https://japan-company-data.p.rapidapi.com/hojin/v1/hojin/{corporateid}/patent"
    querystring = {'X-hojinInfo-api-token': x_hojininfo_api_token, 'Accept': accept, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "japan-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_name(x_hojininfo_api_token: str, accept: str, name: str, limit: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to search for a Japanese company by name, such as "間宮""
    
    """
    url = f"https://japan-company-data.p.rapidapi.com/hojin/v1/hojin"
    querystring = {'X-hojinInfo-api-token': x_hojininfo_api_token, 'Accept': accept, 'name': name, 'limit': limit, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "japan-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_corporate_id(x_hojininfo_api_token: str, accept: str, corporateid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to search by Corporate ID, i.e. 4030001044808"
    
    """
    url = f"https://japan-company-data.p.rapidapi.com/hojin/v1/hojin/{corporateid}"
    querystring = {'X-hojinInfo-api-token': x_hojininfo_api_token, 'Accept': accept, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "japan-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

