import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_geo_location_of_ip_address(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pass IP address to get its geo location and other details"
    ip: Send a proper IPV4 .If valid IP will return result.
        
    """
    url = f"https://mocking-rock.p.rapidapi.com/country/getipgeodata"
    querystring = {'ip': ip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mocking-rock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_locales_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use this API to get the list of available locales 
		Which can be used to generate profile APIs response."
    
    """
    url = f"https://mocking-rock.p.rapidapi.com/profile/known_locales"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mocking-rock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sample_businessprofile(locale: str='en', count: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this API to get fake-generated Business Profile sample data.
		By default, it will generate "en"  locale for a single count.
		If count query is greater than 1 it will randomize the generated profiles 
		to a different locale. 
		To set a default locale pass it as the query.List of the locale you can get from locale API.
		In case of the wrong locale is given it will return the result based on default settings as mentioned above."
    locale: Pass locale to set language and regional details of profile generated. For single count by default its \\\"en\\\" . 
        count: Use to get number of profiles . Max count 1000 for now 
        
    """
    url = f"https://mocking-rock.p.rapidapi.com/profile/business_profile"
    querystring = {}
    if locale:
        querystring['locale'] = locale
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mocking-rock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sample_userprofile(count: int=2, locale: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this API to get fake-generated user Profile sample data.
		By default, it will generate "en"  locale for a single count.
		If count query is greater than 1 it will randomize the generated profiles 
		to a different locale. 
		To set a default locale pass it as the query.List of the locale you can get from locale API.
		In case of the wrong locale is given it will return the result based on default settings as mentioned above."
    count: Use to get number of profiles . Max count 1000 for now 
        locale: Pass locale to set language and regional details of profile generated. For single count by default its \"en\" . 
        
    """
    url = f"https://mocking-rock.p.rapidapi.com/profile/user_profile"
    querystring = {}
    if count:
        querystring['count'] = count
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mocking-rock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_country_codes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this API to get list of countries, Their ISO3 & ISO2 codes."
    
    """
    url = f"https://mocking-rock.p.rapidapi.com/country/countrycode"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mocking-rock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_countrywise_city_data(iso3: str='IND', country: str='IND', count: int=2, iso2: str='IN', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this API to get city list of country. You need to send country name or ISO3 or ISO2 code 
		as query to get it. Which you can get it from other API "Country Codes""
    iso3: PASS 3 character long ISO3 code. You  can get it from country Codes API 
        country: PASS  country name . You  can get it from country Codes API 
        count: Pass count value to get maximum number of results to be returned if available 
        iso2: PASS 2 character long ISO2 code. You  can get it from country Codes API 
        
    """
    url = f"https://mocking-rock.p.rapidapi.com/country/countrywise"
    querystring = {}
    if iso3:
        querystring['iso3'] = iso3
    if country:
        querystring['country'] = country
    if count:
        querystring['count'] = count
    if iso2:
        querystring['iso2'] = iso2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mocking-rock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_data(count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of citys.
		Count limit should be less the 1000"
    count: If not passed by default 10 will be considered
Maximum can be 1000 for now.  
        
    """
    url = f"https://mocking-rock.p.rapidapi.com/country/city"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mocking-rock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

