import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_search(postal_codes: str, website: str=None, bedrooms: int=None, agency: bool=None, furnished: bool=None, property_type: str='House', limit: int=None, city: str=None, search_type: str='rent', price_min: int=None, price_max: int=None, rooms: int=None, has_photos: bool=None, area_max: int=None, since_id: str=None, is_new: bool=None, area_min: int=None, before_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get_search"
    postal_codes: Liste des codes postaux : séparés par une virgule. e.g. \"75003,75001,75004,63122\"
        website: Portail immobilier
        bedrooms: Nombre de chambre
        agency: true: agence  |  false: particulier
        furnished: Bien meublé
        property_type: Type de bien
        limit: Limite d'annonces renvoyées
        city: Ville : les annonces retournées seront une jointure de la ville et du code postal
        search_type: Achat ou location
        price_min: Prix minimum
        price_max: Prix maximum
        rooms: Nombre de pièces
        area_max: Surface maximum
        since_id: Since unique_id
        is_new: Bien neuf ou en construction
        area_min: Surface minimum
        before_id: Before unique_id
        
    """
    url = f"https://fluximmo.p.rapidapi.com/v1/adverts/search"
    querystring = {'postal_codes': postal_codes, }
    if website:
        querystring['website'] = website
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if agency:
        querystring['agency'] = agency
    if furnished:
        querystring['furnished'] = furnished
    if property_type:
        querystring['property_type'] = property_type
    if limit:
        querystring['limit'] = limit
    if city:
        querystring['city'] = city
    if search_type:
        querystring['search_type'] = search_type
    if price_min:
        querystring['price_min'] = price_min
    if price_max:
        querystring['price_max'] = price_max
    if rooms:
        querystring['rooms'] = rooms
    if has_photos:
        querystring['has_photos'] = has_photos
    if area_max:
        querystring['area_max'] = area_max
    if since_id:
        querystring['since_id'] = since_id
    if is_new:
        querystring['is_new'] = is_new
    if area_min:
        querystring['area_min'] = area_min
    if before_id:
        querystring['before_id'] = before_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fluximmo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_alert_adverts(alert_id: int, limit: int=None, since_id: str=None, before_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    alert_id: The unique id of the script
        limit: Limite d'annonces renvoyées
        since_id: Since unique_id
        before_id: Before unique_id
        
    """
    url = f"https://fluximmo.p.rapidapi.com/v1/alerts/{alert_id}/adverts"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if since_id:
        querystring['since_id'] = since_id
    if before_id:
        querystring['before_id'] = before_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fluximmo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_alerts_list_or_create(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fluximmo.p.rapidapi.com/v1/alerts/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fluximmo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_webhook_sample(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fluximmo.p.rapidapi.com/v1/alerts/sample_webhook"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fluximmo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_annonce(unique_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fluximmo.p.rapidapi.com/v1/adverts/{unique_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fluximmo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_annonce_by_urls(url: str, source: str=None, site_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    url: url with https://
        source: Source
        site_id: Site ID
        
    """
    url = f"https://fluximmo.p.rapidapi.com/v1/adverts/urls"
    querystring = {'url': url, }
    if source:
        querystring['source'] = source
    if site_id:
        querystring['site_id'] = site_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fluximmo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_an_alert(alert_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    alert_id: The unique id of the script
        
    """
    url = f"https://fluximmo.p.rapidapi.com/v1/alerts/{alert_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fluximmo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_portail_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fluximmo.p.rapidapi.com/v1/portals/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fluximmo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hello_world(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fluximmo.p.rapidapi.com/v1/ping/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fluximmo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

