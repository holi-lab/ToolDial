import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def flight_booking_software(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Flightsreservationsystem provides Flight Booking Software, Air Booking Engine, Air Ticket System to the global travel industries. For more details, please visit our website: https://www.flightsreservationsystem.com/flight-booking-software"
    
    """
    url = f"https://flight-booking-software1.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-booking-software1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

