import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def prayer_times_for_a_location(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    location: Name of the location where user is at or his state name or his country name or with his latitude and longitude .
        
    """
    url = f"https://muslimsalat.p.rapidapi.com/{location}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "muslimsalat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prayer_times(times: str='daily', date: str='11-07-2013', daylight: bool=None, method: str='5', location: str='london', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will fetch the prayer time based on the location you request for with the given parameters. All following parameters are optional, you can set as you wish. If you wish to set two or more parameters, then you have to set url in the order the parameter is given below."
    times: Limit the prayer times by the value. Daily, weekly, monthly or yearly
        date: Get the prayer times for the given date, please make sure the date is further head or current date. Heads up! Previous dates will be deprecated from the api.
        daylight: Daylight saving for the user, if true then hours are incremented by 1+
        method: Method to use for calculation of the timing. If method is provided invalid based on the country, it will give incorrect timing.   1 = Egyptian General Authority of Survey  2 = University Of Islamic Sciences, Karachi (Shafi)  3 = University Of Islamic Sciences, Karachi (Hanafi)  4 = Islamic Circle of North America  5 = Muslim World League  6 = Umm Al-Qura  7 = Fixed Isha
        location: Name of the location where user is at or his state name or his country name or with his latitude and longitude .
        
    """
    url = f"https://muslimsalat.p.rapidapi.com/(location)/(times)/(date)/(daylight)/(method).json"
    querystring = {}
    if times:
        querystring['times'] = times
    if date:
        querystring['date'] = date
    if daylight:
        querystring['daylight'] = daylight
    if method:
        querystring['method'] = method
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "muslimsalat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

