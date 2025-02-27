import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getprediction(sign: str, period: str='day', day: str='today', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Prediction Zodia Sign from day"
    sign: Sign name <code>e.g.: aries</code>
        period: Period timeframe <code>e.g.: week</code>
        day: Day timeframe <code>e.g.: tomorrow</code>
        lang: Language Code for translation
        
    """
    url = f"https://horoskopos.p.rapidapi.com/zodiac-signs/prediction"
    querystring = {'sign': sign, }
    if period:
        querystring['period'] = period
    if day:
        querystring['day'] = day
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoskopos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnow(timezone: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Zodiac Sign of the current day"
    timezone: Current timezone for the date time needed (check the list https://www.php.net/manual/en/timezones.php)
        lang: Language Code for translation
        
    """
    url = f"https://horoskopos.p.rapidapi.com/zodiac-signs/now"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoskopos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getelementbysign(sign: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get element associated with a Chinese zodiac sign"
    sign: Chinese zodiac sign <code>e.g.: ox</code>
        lang: Language Code for translation
        
    """
    url = f"https://horoskopos.p.rapidapi.com/chinese-zodiac/element-by-sign"
    querystring = {'sign': sign, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoskopos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompatibility(sign1: str=None, lang: str='en', date2: str=None, sign2: str=None, date1: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Compatibility Zodia Sign from date"
    sign1: Sign 1 name <code>e.g.: aries</code>
        lang: Language Code for translation
        date2: Complete Date format <code>e.g.: 1979-12-06</code>
        sign2: Sign 2 name <code>e.g.: sagittarius</code>
        date1: Complete Date format <code>e.g.: 1987-04-06</code>
        
    """
    url = f"https://horoskopos.p.rapidapi.com/zodiac-signs/compatibility"
    querystring = {}
    if sign1:
        querystring['sign1'] = sign1
    if lang:
        querystring['lang'] = lang
    if date2:
        querystring['date2'] = date2
    if sign2:
        querystring['sign2'] = sign2
    if date1:
        querystring['date1'] = date1
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoskopos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsearch(month: int=None, lang: str='en', date: str=None, year: int=None, day: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Zodia Sign from date"
    month: Month of the date <code>e.g.: 4</code>
        lang: Language Code for translation
        date: Complete Date format <code>e.g.: 1987-04-06</code>
        year: Year of the date <code>e.g.: 2021</code>
        day: Day of the date <code>e.g.: 6</code>
        
    """
    url = f"https://horoskopos.p.rapidapi.com/zodiac-signs/search"
    querystring = {}
    if month:
        querystring['month'] = month
    if lang:
        querystring['lang'] = lang
    if date:
        querystring['date'] = date
    if year:
        querystring['year'] = year
    if day:
        querystring['day'] = day
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoskopos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcurrent(timezone: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Zodiac Sign of the current day"
    timezone: Current timezone for the date time needed (check the list https://www.php.net/manual/en/timezones.php)
        lang: Language Code for translation
        
    """
    url = f"https://horoskopos.p.rapidapi.com/zodiac-signs/"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoskopos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettoday(lang: str='en', timezone: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Zodiac Sign of the current day"
    lang: Language Code for translation
        timezone: Current timezone for the date time needed (check the list https://www.php.net/manual/en/timezones.php)
        
    """
    url = f"https://horoskopos.p.rapidapi.com/zodiac-signs/today"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoskopos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

