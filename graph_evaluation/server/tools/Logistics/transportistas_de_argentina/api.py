import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tracking_correo_argentino_result_task_task_id(task_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Result for one Task ID."
    task_id: Task ID
        
    """
    url = f"https://transportistas-de-argentina.p.rapidapi.com/tracking/correo_argentino/result_task/{task_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transportistas-de-argentina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_states_stateisocode(stateisocode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of city for iso state."
    stateisocode: State ISO Code
        
    """
    url = f"https://transportistas-de-argentina.p.rapidapi.com/cities/states/{stateisocode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transportistas-de-argentina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_postcode_stateisocode_postcode(stateisocode: str, postcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of city for iso state and postcode."
    stateisocode: State ISO Code
        postcode: Postcode
        
    """
    url = f"https://transportistas-de-argentina.p.rapidapi.com/cities/postcode/{stateisocode}/{postcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transportistas-de-argentina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_search_stateisocode_keyword(stateisocode: str, keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search city for iso state and keyword name."
    stateisocode: State ISO Code
        keyword: Keyword to search, example: Caballito
        
    """
    url = f"https://transportistas-de-argentina.p.rapidapi.com/cities/search/{stateisocode}/{keyword}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transportistas-de-argentina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_states(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of states for Argentina with iso code."
    
    """
    url = f"https://transportistas-de-argentina.p.rapidapi.com/cities/states"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transportistas-de-argentina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quotes_city_correo_argentino_weight_stateisocodesrc_normalizecitynamesrc_stateisocodedst_normalizecitynamedst(normalizecitynamesrc: str, stateisocodedst: str, normalizecitynamedst: str, weight: int, stateisocodesrc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of quote for iso state and city keyword name in Correo Argentino."
    normalizecitynamesrc: Normalize City Name of Source
        stateisocodedst: State ISO Code of Destination
        normalizecitynamedst: Normalize City Name of Destination
        weight: Weight in KG
        stateisocodesrc: State ISO Code of Source
        
    """
    url = f"https://transportistas-de-argentina.p.rapidapi.com/quotes/city/correo_argentino/{weight}/{stateisocodesrc}/{normalizecitynamesrc}/{stateisocodedst}/{normalizecitynamedst}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transportistas-de-argentina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quotes_postcode_oca_cuit_operativa_cost_weight_volume_postcodesrc_postcodedst(postcodedst: int, cuit: str, operativa: str, cost: int, postcodesrc: int, volume: int, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quote for postcode in OCA e-Pack."
    postcodedst: Postcode Destination
        cuit: CUIT of your account in OCA e-Pack
        operativa: Operativa number of your account in OCA e-Pack
        cost: Cost of products in ARS
        postcodesrc: Postcode Source
        volume: Volume in cm3
        weight: Weight in KG
        
    """
    url = f"https://transportistas-de-argentina.p.rapidapi.com/quotes/postcode/oca/{cuit}/{operativa}/{cost}/{weight}/{volume}/{postcodesrc}/{postcodedst}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transportistas-de-argentina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quotes_postcode_correo_argentino_weight_postcodesrc_postcodedst(postcodedst: int, postcodesrc: int, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of quote for postcode in Correo Argentino."
    postcodedst: Postcode Destination
        postcodesrc: Postcode Source
        weight: Weight in KG
        
    """
    url = f"https://transportistas-de-argentina.p.rapidapi.com/quotes/postcode/correo_argentino/{weight}/{postcodesrc}/{postcodedst}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transportistas-de-argentina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tracking_correo_argentino_create_task_service_tracking_code(service: str, tracking_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create task to get the history.
		The result of the trace can be obtained after approximately 20-60 seconds by calling the endpoint: /tracking/correo_argentino/result_task/:task_id
		**IMPORTANT: ** The result will only be stored for approx 1 hour."
    service: Service: ecommerce, mercadolibre, national, national-plus or national-international
        tracking_code: Tracking code
        
    """
    url = f"https://transportistas-de-argentina.p.rapidapi.com/tracking/correo_argentino/create_task/{service}/{tracking_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transportistas-de-argentina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def offices_postcode_service_postcode(service: str, postcode: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of offices for iso state and postcode."
    service: Service Name: correo_argentino, oca, andreani
        postcode: Postcode
        
    """
    url = f"https://transportistas-de-argentina.p.rapidapi.com/offices/postcode/{service}/{postcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transportistas-de-argentina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def offices_search_service_stateisocode_keyword(keyword: str, service: str, stateisocode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of offices for iso state and postcode."
    keyword: Keyword to search, example: Caballito
        service: Service Name: correo_argentino, oca, andreani
        stateisocode: State ISO Code
        
    """
    url = f"https://transportistas-de-argentina.p.rapidapi.com/offices/search/{service}/{stateisocode}/{keyword}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transportistas-de-argentina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

