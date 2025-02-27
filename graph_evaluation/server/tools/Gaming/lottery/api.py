import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def millipiyango(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Service that brings you the full list of Milli Piyango."
    date: Enter the date you would like to know the results (Format: 'YYYYMMDD')
        
    """
    url = f"https://lottery2.p.rapidapi.com/millipiyango"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def onnumara(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Service bringing numerical On Numara results."
    
    """
    url = f"https://lottery2.p.rapidapi.com/onNumara"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sanstopu(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Service bringing numerical Şans Topu results."
    
    """
    url = f"https://lottery2.p.rapidapi.com/sanstopu"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sayisalloto(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Service bringing Sayısal Loto results."
    
    """
    url = f"https://lottery2.p.rapidapi.com/sayisalLoto"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def superloto(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Service bringing numerical Super Loto results."
    
    """
    url = f"https://lottery2.p.rapidapi.com/superLoto"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def euromillions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information on the last draw of the Euro Millions lottery game returning service."
    
    """
    url = f"https://lottery2.p.rapidapi.com/euroMillions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def euromillionschecker(numbers: str, ls: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For the last draw of the Euro Millions game of chance; Ticket check service showing the winnings according to the entered ticket."
    numbers: Numbers of tickets other than lucky stars.
        ls: Numbers of tickets lucky stars numbers.
        
    """
    url = f"https://lottery2.p.rapidapi.com/euroMillionsChecker"
    querystring = {'numbers': numbers, 'ls': ls, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usamegamillions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information on the last draw of the USA Mega Millions lottery game returning service."
    
    """
    url = f"https://lottery2.p.rapidapi.com/usaMegaMillions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usamegamillionschecker(mb: str, numbers: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For the last draw of the United States Mega Millions game of chance; Ticket check service showing the winnings according to the entered ticket."
    mb: Numbers of tickets megaball. (etc: 6)
        numbers: Numbers of tickets other than megaball. (etc: 12,13,20,22,25)
        
    """
    url = f"https://lottery2.p.rapidapi.com/usaMegaMillionsChecker"
    querystring = {'mb': mb, 'numbers': numbers, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usapowerball(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information on the last draw of the USA Powerball lottery game returning service."
    
    """
    url = f"https://lottery2.p.rapidapi.com/usaPowerball"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usapowerballchecker(pb: str, numbers: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For the last draw of the United States Powerball game of chance; Ticket check service showing the winnings according to the entered ticket."
    pb: Numbers of tickets powerball. (etc: 6)
        numbers: Numbers of tickets other than powerball. (etc: 12,13,20,22,25)
        
    """
    url = f"https://lottery2.p.rapidapi.com/usaPowerballChecker"
    querystring = {'pb': pb, 'numbers': numbers, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usasuperlottoplus(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information on the last draw of the USA SuperLotto Plus lottery game returning service."
    
    """
    url = f"https://lottery2.p.rapidapi.com/usaSuperLottoPlus"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usasuperlottopluschecker(mn: str, numbers: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For the last draw of the United States Super Lotto Plus game of chance; Ticket check service showing the winnings according to the entered ticket."
    mn: Numbers of tickets powerball. (etc: 6)
        numbers: Numbers of tickets other than powerball. (etc: 12,13,20,22,25)
        
    """
    url = f"https://lottery2.p.rapidapi.com/usaSuperLottoPlusChecker"
    querystring = {'mn': mn, 'numbers': numbers, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lottery2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

