import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def postal_code_by_province_slug(province_slug: str, perpage: int=25, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of area and postal code by Province."
    perpage: Default max result per page, max 25
        page: Page number of the result
        
    """
    url = f"https://indonesian-postal-code-list-and-search.p.rapidapi.com/kodepos/provinsi/{province_slug}"
    querystring = {}
    if perpage:
        querystring['perPage'] = perpage
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesian-postal-code-list-and-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_data_by_query(keyword: str, perpage: str='25', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can query data from our database by some keyword such as:
		
		- Province Name (Provinsi)
		- City Name (Kota / Kabupaten)
		- Sub District Name (Kecamatan)
		- Urban Name (Kelurahan/Desa)
		-  Postal Code (Kode Pos)"
    
    """
    url = f"https://indonesian-postal-code-list-and-search.p.rapidapi.com/search"
    querystring = {'keyword': keyword, }
    if perpage:
        querystring['perPage'] = perpage
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesian-postal-code-list-and-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_postal_code_information(postal_code: str, perpage: str='25', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details from some postal code, sometime some postal code used by one or more urban / village"
    
    """
    url = f"https://indonesian-postal-code-list-and-search.p.rapidapi.com/kodepos/{postal_code}"
    querystring = {}
    if perpage:
        querystring['perPage'] = perpage
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesian-postal-code-list-and-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def postal_code_by_urban_kelurahan_desa(urban_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get postal code from an urban / village (Kelurahan/Desa)"
    
    """
    url = f"https://indonesian-postal-code-list-and-search.p.rapidapi.com/kodepos/kelurahan/{urban_slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesian-postal-code-list-and-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def postal_code_by_sub_district_kecamatan(sub_district_slug: str, perpage: str='25', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get postal code from selected Sub District (Kecamatan)"
    
    """
    url = f"https://indonesian-postal-code-list-and-search.p.rapidapi.com/kodepos/kecamatan/{sub_district_slug}"
    querystring = {}
    if perpage:
        querystring['perPage'] = perpage
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesian-postal-code-list-and-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def postal_code_by_city_slug(city_slug: str, page: str='1', perpage: str='25', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get postal code list from area on selected city"
    
    """
    url = f"https://indonesian-postal-code-list-and-search.p.rapidapi.com/kodepos/kota/{city_slug}"
    querystring = {}
    if page:
        querystring['page'] = page
    if perpage:
        querystring['perPage'] = perpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesian-postal-code-list-and-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_indonesian_province(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This list will be used ond another endpoint to take postal code by province name"
    
    """
    url = f"https://indonesian-postal-code-list-and-search.p.rapidapi.com/provinsi"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesian-postal-code-list-and-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

