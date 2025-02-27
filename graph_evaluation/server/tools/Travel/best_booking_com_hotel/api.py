import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_best_booking_com_accommodation(countryname: str, cityname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "find best booking.com accommodation"
    
    """
    url = f"https://best-booking-com-hotel.p.rapidapi.com/booking/best-accommodation"
    querystring = {'countryName': countryname, 'cityName': cityname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "best-booking-com-hotel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

