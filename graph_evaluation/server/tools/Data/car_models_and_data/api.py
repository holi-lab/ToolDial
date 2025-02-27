import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_car_by_id(is_id: int, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a car by id. This route will show the complete car data. General data and also technical informations.
		
		You can also select one of three languages. Currently supported languages are German (de)(default), English (en) and France (fr)"
    lang: You can use one of three languages. 
German => de (default, fallback)
English => en
France => fr
The description of the technical details will be changed to the language
        
    """
    url = f"https://car-models-and-data.p.rapidapi.com/api/v1/cars/{is_id}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-models-and-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_cars_by_hsn_tsn(hsn: str, tsn: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a car by hsn and tsn. Only vehicles registered in Germany have an HSN/TSN at all."
    
    """
    url = f"https://car-models-and-data.p.rapidapi.com/api/v1/cars/hsn/{hsn}/tsn/{tsn}"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-models-and-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_model_series_generation_by_name(page: int=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows you all model series generation containing a certain text"
    
    """
    url = f"https://car-models-and-data.p.rapidapi.com/api/v1/model-series/generations/find"
    querystring = {}
    if page:
        querystring['page'] = page
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-models-and-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_model_series_by_name(name: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show you all model series containing a certain string"
    
    """
    url = f"https://car-models-and-data.p.rapidapi.com/api/v1/model-series/find"
    querystring = {}
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-models-and-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def showing_a_list_of_cars(page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list with all cars. These are provided per page. 20 cars are displayed on one page."
    
    """
    url = f"https://car-models-and-data.p.rapidapi.com/api/v1/cars"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-models-and-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def show_all_brands(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows a List of all car brands"
    
    """
    url = f"https://car-models-and-data.p.rapidapi.com/api/v1/brands"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-models-and-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_cars_by_maker_name_or_year(date: str=None, maker: str='vw', name: str='golf', page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find cars by maker, name or model year in a paginated list. The format for the date is month/year for example: 05/21 is mai 2021."
    
    """
    url = f"https://car-models-and-data.p.rapidapi.com/api/v1/cars/find"
    querystring = {}
    if date:
        querystring['date'] = date
    if maker:
        querystring['maker'] = maker
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-models-and-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def show_model_series_generations(page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show a list of model series generations. Per page are 200 results listed. The generations also include the model series and brands."
    
    """
    url = f"https://car-models-and-data.p.rapidapi.com/api/v1/model-series/generations"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-models-and-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def show_car_model_series(page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of car model series"
    
    """
    url = f"https://car-models-and-data.p.rapidapi.com/api/v1/model-series"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-models-and-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_brand_by_name(name: str='sko', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the brand you need by the name"
    
    """
    url = f"https://car-models-and-data.p.rapidapi.com/api/v1/brands"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-models-and-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

