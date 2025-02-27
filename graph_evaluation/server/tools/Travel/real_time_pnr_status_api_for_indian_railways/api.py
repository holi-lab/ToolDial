import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pnr_status(pnr_status: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is an unofficial PNR Status API that provides real-time information on the status of Indian Railways train reservations. This API is designed for educational purposes only and is not affiliated with Indian Railways in any way."
    
    """
    url = f"https://real-time-pnr-status-api-for-indian-railways.p.rapidapi.com/indianrail/{pnr_status}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-pnr-status-api-for-indian-railways.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_train_running_status(number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Live Train Running Status is a crucial keyword in the world of railway transportation, and Qrail is one of the leading providers of this service. With Live Train Running Status, passengers can get real-time updates on the current location and running status of their train, including expected arrival and departure times. Qrail's Live Train Running Status service is highly reliable and accurate, providing passengers with peace of mind and ensuring that they arrive at their destinations on time. With this service, passengers can plan their journey better and avoid any inconvenience caused by unexpected delays or cancellations."
    
    """
    url = f"https://real-time-pnr-status-api-for-indian-railways.p.rapidapi.com/trainman/{number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-pnr-status-api-for-indian-railways.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

