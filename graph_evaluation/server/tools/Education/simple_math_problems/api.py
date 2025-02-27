import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_addition_problems(digits: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves 10 addition problems, with one correct answer and three incorrect answers. The additional parameter specifies if the problem includes single digit numbers using 'single', double digit numbers using 'double', or one of either using 'singledouble'."
    
    """
    url = f"https://simple-math-problems.p.rapidapi.com/addition/{digits}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-math-problems.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_division_problems(digits: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves 10 division problems, with one correct answer and three incorrect answers. The additional parameter specifies if the problem includes single digit numbers using 'single', double digit numbers using 'double', or one of either using 'singledouble'."
    
    """
    url = f"https://simple-math-problems.p.rapidapi.com/division/{digits}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-math-problems.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_multiplication_problems(digits: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves 10 multiplication problems, with one correct answer and three incorrect answers. The additional parameter specifies if the problem includes single digit numbers using 'single', double digit numbers using 'double', or one of either using 'singledouble'."
    
    """
    url = f"https://simple-math-problems.p.rapidapi.com/multiplication/{digits}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-math-problems.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_subtraction_problems(digits: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves 10 subtraction problems, with one correct answer and three incorrect answers. The additional parameter specifies if the problem includes single digit numbers using 'single', double digit numbers using 'double', or one of either using 'singledouble'."
    
    """
    url = f"https://simple-math-problems.p.rapidapi.com/subtraction/{digits}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-math-problems.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

