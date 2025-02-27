import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def drawings_between_dates(date2: str, date1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes in two dates and gives back all the winning numbers in between."
    
    """
    url = f"https://mega-millions.p.rapidapi.com/BetweenDates/{date1}/{date2}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-millions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def general_statistics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint pulls  statistics of the Mega Millions numbers. For each ball (denoted firstNumber, secondNumber, thirdNumber, fourthNumber, fifthNumber, megaBall, megaplier) of the min, max, mean, median, mode, standard deviation, variance, sample variance, skewness, and kurtosis is calculated. Lastly, there is a frequency of occurrence list that shows how often each number has appeared in the result set."
    
    """
    url = f"https://mega-millions.p.rapidapi.com/stats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-millions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_drawing(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pull the latest Mega Millions winning numbers."
    
    """
    url = f"https://mega-millions.p.rapidapi.com/latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-millions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_ticket(fifthnumber: int, secondnumber: int, megaball: int, thirdnumber: int, fouthnumber: int, date: str, firstnumber: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Let's see if your ticket is a winner! This endpoint will look at the date of the requested drawing and the numbers on your ticket to determine if you are a winner."
    
    """
    url = f"https://mega-millions.p.rapidapi.com/CheckTicket/{date}/{firstnumber}/{secondnumber}/{thirdnumber}/{fouthnumber}/{fifthnumber}/{megaball}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-millions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_drawings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Mega Millions Winning Numbers"
    
    """
    url = f"https://mega-millions.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-millions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_10_drawings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pull the latest 10 winning numbers"
    
    """
    url = f"https://mega-millions.p.rapidapi.com/latest10"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-millions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_quickpick_for_play(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate random numbers for picking the Mega Millions lottery numbers."
    
    """
    url = f"https://mega-millions.p.rapidapi.com/QuickPick"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-millions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def drawing_by_date(drawingdate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows the user to find the information on one drawing by using the data of the drawing in the yyyy-MM-dd format."
    
    """
    url = f"https://mega-millions.p.rapidapi.com/{drawingdate}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-millions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

