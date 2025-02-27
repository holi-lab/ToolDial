import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v1_user_id(is_id: int, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fakeapi2.p.rapidapi.com/api/v1/user/{is_id}"
    querystring = {}
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fakeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_tmp_type_id(type: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fakeapi2.p.rapidapi.com/api/v1/tmp/{type}/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fakeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_from_now(datetimeutc: str=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fakeapi2.p.rapidapi.com/time-from-now"
    querystring = {}
    if datetimeutc:
        querystring['dateTimeUtc'] = datetimeutc
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fakeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_difference(locale: str=None, comparetodatetimeutc: str=None, datetimeutc: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fakeapi2.p.rapidapi.com/time-difference "
    querystring = {}
    if locale:
        querystring['locale'] = locale
    if comparetodatetimeutc:
        querystring['compareToDateTimeUtc'] = comparetodatetimeutc
    if datetimeutc:
        querystring['dateTimeUtc'] = datetimeutc
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fakeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seconds(seconds: int=None, precision: int=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fakeapi2.p.rapidapi.com/seconds"
    querystring = {}
    if seconds:
        querystring['seconds'] = seconds
    if precision:
        querystring['precision'] = precision
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fakeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_user_id_orders(is_id: int, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fakeapi2.p.rapidapi.com/api/user/{is_id}/orders"
    querystring = {}
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fakeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_user(page: int=0, locale: str='en', itemsinpage: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fakeapi2.p.rapidapi.com/api/v1/user"
    querystring = {}
    if page:
        querystring['page'] = page
    if locale:
        querystring['locale'] = locale
    if itemsinpage:
        querystring['itemsInPage'] = itemsinpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fakeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hours(hours: int=None, precision: int=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fakeapi2.p.rapidapi.com/hours"
    querystring = {}
    if hours:
        querystring['hours'] = hours
    if precision:
        querystring['precision'] = precision
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fakeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_fake_data_from_template(is_id: int=None, locale: str='en', handlebartemplate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fakeapi2.p.rapidapi.com/api/fake-data-from-template"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if locale:
        querystring['locale'] = locale
    if handlebartemplate:
        querystring['handlebarTemplate'] = handlebartemplate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fakeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_number_to_words_number(number: int, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fakeapi2.p.rapidapi.com/api/number-to-words/{number}"
    querystring = {}
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fakeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_tmp_type(type: str, page: int=0, itemsinpage: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fakeapi2.p.rapidapi.com/api/v1/tmp/{type}"
    querystring = {}
    if page:
        querystring['page'] = page
    if itemsinpage:
        querystring['itemsInPage'] = itemsinpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fakeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def minutes(precision: int=None, locale: str=None, minutes: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://fakeapi2.p.rapidapi.com/minutes"
    querystring = {}
    if precision:
        querystring['precision'] = precision
    if locale:
        querystring['locale'] = locale
    if minutes:
        querystring['minutes'] = minutes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fakeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

