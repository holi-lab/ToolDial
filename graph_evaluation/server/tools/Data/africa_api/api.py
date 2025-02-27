import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tourism_per_country(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns best tourism locations and categories you can explore in a country in Africa
		Takes parameter <country_name>"
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/tourism"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def independence_all_countries(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If you provide the name of a country as a parameter, the api will return information about its independence date and the country or countries that colonized it.
		Takes Parameter: <county_name> with value: <all>
		Response example: 
		{
		"country":"Kenya"
		"colonizer":"United Kingdom"
		"independence_date":"December 12, 1963"
		}"
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/independence"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages_per_country(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a json of all languages spoken in a specific country in Africa
		Takes parameter <country_name> 
		Response example:
		{2 items
		"other_languages":[12 items
		0:
		"English"
		1:
		"Kikuyu"
		2:
		"Luhya"
		3:
		"Luo"
		4:
		"Kalenjin"
		5:
		"Kamba"
		6:
		"Meru"
		7:
		"Embu"
		8:
		"Maasai"
		9:
		"Turkana"
		10:
		"Samburu"
		11:
		"Somali"
		]
		"national_language":"Swahili"
		}"
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/languages"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sizes_all_countries(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the sizes of all countries in Africa
		Takes parameter: <country_name> with value <all>
		Response example:
		[54 items
		0:{3 items
		"size":2381741
		"unit":"km2"
		"country":"Algeria"
		}
		]"
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/sizes"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resources_per_country(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives you a list of the most important resources that an African country relies on to keep their economy running.
		Takes parameter <country_name>"
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/resources"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_allcontries(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a json of all countries in Africa. Details include: "name":"Algeria"
		"latitude", "longitude", "phone_code", "abbreviation", "capital_city""
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/countries"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tourism_allcountries(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns best tourism locations and categories you can explore in each country in Africa"
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/tourism"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages_allcountries(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a json of all languages spoken in each country in Africa
		Takes parameter <country_name> with value <all>
		Response example:
		{54 items
		"Chad":{2 items
		"other_languages":[11 items
		0:
		"Sara"
		1:
		"Maba"
		2:
		"Kanembu"
		3:
		"Kotoko"
		4:
		"Bagirmi"
		5:
		"Boulala"
		6:
		"Gorane"
		7:
		"Marba"
		8:
		"Masalit"
		9:
		"Tama"
		10:
		"Teda"
		]
		"national_language":[2 items
		0:
		"French"
		1:
		"Arabic"
		]
		}
		]"
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/languages"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sizes_percountry(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the size of a specific country.
		Takes parameter: <country_name>
		Response example:
		{
		"size":580367
		"unit":"km2"
		"country":"Kenya"
		}"
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/sizes"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def province_state_county_percountry(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns json for province/state/county in a specific country in Africa"
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/counties"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def independence_per_country(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If you provide the name of a country as a parameter, the api will return information about its independence date and the country or countries that colonized it.
		Takes Parameter: <county_name>"
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/independence"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resources_allcontries(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives you a list of the most important resources that each African country relies on to keep their economy running."
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/resources"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_specific_country(country_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a json of specific country in Africa eg Kenya. Details include: "name":"Algeria"
		"latitude", "longitude", "phone_code", "abbreviation", "capital_city""
    
    """
    url = f"https://africa-api.p.rapidapi.com/africa/api/v1/africa/countries"
    querystring = {'country_name': country_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "africa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

