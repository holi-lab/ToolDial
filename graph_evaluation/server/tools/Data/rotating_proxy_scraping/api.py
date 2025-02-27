import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_active_zones(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get active zones"
    
    """
    url = f"https://rotating-proxy-scraping.p.rapidapi.com/api/zone/get_active_zones"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rotating-proxy-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_available_data_center_ips_per_zone(zone: str, country: str='il', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the available data center IPs per zone"
    country: br,jp,th,by,nl,ua,uz,ge,il,md,kg,az,tj,lv,tm,lt,tw,sg,kr,ma,pa,pl,pt,us,ca,sa,ae,my,cn,ie,se,cz,nz,fi,ph,dk,ch,no,in,cl,pe,ar,vn,lk,ec,hu,at,la,bg,kz,gb,bd,al,ee,de,es,gr,it,sk,am,hk,eg,pk,bo,ru,lu,tr,kh,id,fr,ro,co,za,au,do,jm,jo,hr,cy,is,cr,im,be,mx
        
    """
    url = f"https://rotating-proxy-scraping.p.rapidapi.com/api/zone/route_ips"
    querystring = {'zone': zone, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rotating-proxy-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_usage(page: int=1, skip: int=0, limit: int=10, is_from: str=None, to: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get usage"
    
    """
    url = f"https://rotating-proxy-scraping.p.rapidapi.com/api/zone/usage"
    querystring = {}
    if page:
        querystring['page'] = page
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rotating-proxy-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_zone_static_datacenter_ips(zone: str, ip_per_country: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get zone static Datacenter IPs"
    
    """
    url = f"https://rotating-proxy-scraping.p.rapidapi.com/api/zone/ips"
    querystring = {'zone': zone, }
    if ip_per_country:
        querystring['ip_per_country'] = ip_per_country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rotating-proxy-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

