import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def member_updatesmsstatus(appid: str, appkey: str, union: str, sign: str, kt: str, sms_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Update sms status after people read it"
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'union': union, 'sign': sign, 'kt': kt, 'sms_id': sms_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def grants_getlist(appid: str, appkey: str, union: str, sign: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get grants list"
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'union': union, 'sign': sign, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def apply_getdetail(appid: str, appkey: str, union: str, is_id: str, sign: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details by apply id"
    id: Id of application
        
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'union': union, 'id': is_id, 'sign': sign, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def propety_find(appid: str, appkey: str, sign: str, union: str, size: str='30,60', bed: str='3,6', region: str='south', name: str="billy's home", type: str='one of cottage,house,apartment,hardtop_camper,softtop_camper', attr: str='yes', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find properties according to some searching fields"
    appid: Application id
        size: Find properties area size between 30  and 60
        bed: Find properties have between 3 and 6 sleeping spaces
        region: Find property in specific region
        name: Find property by its name
        type: Type of properties
        attr: Get main list attribute
        
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'sign': sign, 'union': union, }
    if size:
        querystring['_size'] = size
    if bed:
        querystring['_bed'] = bed
    if region:
        querystring['_region'] = region
    if name:
        querystring['_name'] = name
    if type:
        querystring['_type'] = type
    if attr:
        querystring['attr'] = attr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_gettypes(appid: str, appkey: str, sign: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all types of properties"
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'sign': sign, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ticket_find(appid: str, appkey: str, union: str, sign: str, start: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ticket list"
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'union': union, 'sign': sign, }
    if start:
        querystring['start'] = start
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mobile_topnews(appid: str, appkey: str, sign: str, union: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top news for mobile home page"
    appid: Application id
        
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'sign': sign, 'union': union, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def member_getapplications(appid: str, appkey: str, union: str, sign: str, kt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get application history"
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'union': union, 'sign': sign, 'kt': kt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def member_getnewsmsamount(appid: str, appkey: str, union: str, sign: str, kt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get new sms amount"
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'union': union, 'sign': sign, 'kt': kt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def member_verify(appid: str, appkey: str, kt: str, pass: str, sign: str, union: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify a member in specific union"
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'kt': kt, 'pass': pass, 'sign': sign, 'union': union, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def member_getinfo(appid: str, appkey: str, union: str, kt: str, sign: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info of member"
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'union': union, 'kt': kt, 'sign': sign, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coupon_find(appid: str, appkey: str, union: str, sign: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find coupon"
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'union': union, 'sign': sign, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def member_getrentedlist(appid: str, appkey: str, union: str, sign: str, kt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rented order list"
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'union': union, 'sign': sign, 'kt': kt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_getinfo(appid: str='000123', appkey: str='ce82d69cd20geff7e72c47d90dda29fe', is_id: str='123', sign: str='ge82d90cd20geff7e72c47d90dda29fe', union: str='work,fit,ki etc..', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Basic information of a single property"
    appid: Application id
        appkey: Application key
        is_id: Property id
        sign: Sign of request,must be a md5 string
        union: Name of  union
        
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {}
    if appid:
        querystring['appid'] = appid
    if appkey:
        querystring['appkey'] = appkey
    if is_id:
        querystring['id'] = is_id
    if sign:
        querystring['sign'] = sign
    if union:
        querystring['union'] = union
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def member_gettickets(appid: str, appkey: str, union: str, kt: str, sign: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tickets order"
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'union': union, 'kt': kt, 'sign': sign, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_period_getymperiods(sign: str, appid: str, appkey: str, property_id: str, year: str, union: str, month: str='09,10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get periods of a property in a year or a month"
    sign: Sign of request,must be a md5 string
        year: Year
        month: Month
        
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'sign': sign, 'appid': appid, 'appkey': appkey, 'property_id': property_id, 'year': year, 'union': union, }
    if month:
        querystring['month'] = month
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ticket_getorderpayment(appid: str, appkey: str, union: str, sign: str, method: str, order_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get specific ticket info"
    method: name of method
        
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'union': union, 'sign': sign, 'method': method, 'order_id': order_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def member_getsms(appid: str, appkey: str, sign: str, union: str, kt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sms"
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'sign': sign, 'union': union, 'kt': kt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_period_getymexcdays(sign: str, month: str, appid: str, appkey: str, property_id: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    sign: Sign of request,must be a md5 string
        month: Month
        
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'sign': sign, 'month': month, 'appid': appid, 'appkey': appkey, 'property_id': property_id, 'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def caretaker_verify(appid: str, appkey: str, union: str, sign: str, kt: str, pass: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://funlake-getunions.p.rapidapi.com/Server.php"
    querystring = {'appid': appid, 'appkey': appkey, 'union': union, 'sign': sign, 'kt': kt, 'pass': pass, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "funlake-getunions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

