import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_range_of_items_from_fibonacci_list(rightindex: int, leftindex: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint will return an interval with items from Fibonacci list. The inputs for this endpoint are the left and right indexes.
		Let`s consider the Fibonacci items from -6 to 6: 
		Index:      -6    -5    -4    -3    -2    -1    0     1     2     3     4     5     6
		Value:     -8     5     -3     2     -1     1     0     1     1     2     3     5     8   
		The API call /fibonacci-number/-3/2 will return an array containing the elements from index -3 to index 2.  The objects from array has two properties: "index" and "value".\
		The inputs must be a numbers between -10000 and 10000."
    
    """
    url = f"https://fibonacci.p.rapidapi.com/fibonacci-list/{leftindex}/{rightindex}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fibonacci.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_number_from_fibonacci_list(number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint will return a specific number from Fibonacci list. The input for this endpoint is the number index in Fibonacci list.
		Let`s consider the Fibonacci items from -6 to 6: 
		Index:      -6    -5    -4    -3    -2    -1    0     1     2     3     4     5     6
		Value:     -8     5     -3     2     -1     1     0     1     1     2     3     5     8   
		The API call /fibonacci-number/-5 will return 5.
		The API call /fibonacci-number/3 will return 2.
		And so on...
		The input number must be a number between -10000 and 10000."
    
    """
    url = f"https://fibonacci.p.rapidapi.com/fibonacci-number/{number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fibonacci.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

