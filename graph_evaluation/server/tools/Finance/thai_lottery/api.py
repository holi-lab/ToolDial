import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_lottery_list_by_date_optional_2(date: str, fresh: bool=None, is_from: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Lottery Result By Date"
    
    """
    url = f"https://thai-lottery1.p.rapidapi.com/index3"
    querystring = {'date': date, }
    if fresh:
        querystring['fresh'] = fresh
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lottery1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_lottery_list_by_date(date: str, is_from: bool=None, fresh: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Lottery Result By Date"
    is_from: first array (0,0) show a day that is lottery out
        fresh: Delete a Cache file and Update
        
    """
    url = f"https://thai-lottery1.p.rapidapi.com/"
    querystring = {'date': date, }
    if is_from:
        querystring['from'] = is_from
    if fresh:
        querystring['fresh'] = fresh
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lottery1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_thai_lottery_news(count: int, fulldesc: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Thai Lottery News"
    count: set a number to give news amount what you want
        
    """
    url = f"https://thai-lottery1.p.rapidapi.com/lotnews"
    querystring = {'count': count, }
    if fulldesc:
        querystring['fulldesc'] = fulldesc
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lottery1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_last_lottery(info: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Last Lottery"
    info: get a info like date
        
    """
    url = f"https://thai-lottery1.p.rapidapi.com/lastlot"
    querystring = {}
    if info:
        querystring['info'] = info
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lottery1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_lottery_date(format: str='combothtext', cache: str='yes', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All Lottery Date
		format :
		no format = 01112564
		thtext = 01 พฤศจิกายน 2564
		combothtext = ["01112564","01 พฤศจิกายน 2564"]
		*becareful sometime is be a long time loading or not working"
    cache: Use a Cache File
        
    """
    url = f"https://thai-lottery1.p.rapidapi.com/god"
    querystring = {}
    if format:
        querystring['format'] = format
    if cache:
        querystring['cache'] = cache
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lottery1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_date_list_of_this_lottery_is_have_prize(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Date list (from 2550/2007 to now) of this Lottery is Have Prize
		*becareful sometime is be a long time loading or not working"
    
    """
    url = f"https://thai-lottery1.p.rapidapi.com/finddol"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lottery1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_lottery_have_prize_or_not_by_date(by: str, search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check Lottery Have Prize?
		1st prize result is 111111
		first 3 lottery  prize result is 333000
		last 3 lottery prize result is 000333
		last 2 lottery prize result is 000022
		near 1st prize result is 111112
		2nd prize result is 222222
		3rd prize result is 333333
		4rd prize result is 444444
		5rd prize result is 555555"
    
    """
    url = f"https://thai-lottery1.p.rapidapi.com/checklottery"
    querystring = {'by': by, 'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lottery1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_date_of_lottery_by_year(year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all date of Lottery By Year"
    
    """
    url = f"https://thai-lottery1.p.rapidapi.com/gdpy"
    querystring = {'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lottery1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def today_is_thai_loterry_day(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Today is Thai Loterry Day?
		result yes or no
		*after 1st prize is out result is no"
    
    """
    url = f"https://thai-lottery1.p.rapidapi.com/reto"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lottery1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_lottery_list_by_date_optional(date: str, fresh: bool=None, is_from: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Lottery Result By Date"
    
    """
    url = f"https://thai-lottery1.p.rapidapi.com/index2"
    querystring = {'date': date, }
    if fresh:
        querystring['fresh'] = fresh
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lottery1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_image_of_lucky_number(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Image of Lucky number"
    
    """
    url = f"https://thai-lottery1.p.rapidapi.com/getchit"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lottery1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

