import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_airbnb_income_history(bedrooms: int, bathrooms: int, countrycode: str, coordinate: str, capacity: int=None, radiusinmeter: int=5000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Input
		- coordinate (Lat, Lng of a property)
		- countryCode (Country code in ISO Alpha-2 format. E.g. US, GB, KR, MX etc )
		- bedrooms (Number of bedrooms) 
		- radiusInMeter (The API will only use the listings within this radius) 
		
		List of country codes: https://www.nationsonline.org/oneworld/country_code_list.htm
		
		Output: 
		- Number of sample: number of listings used to calculate the result. 
		- average: This object contains average of key metrics
		- percentile25
		- percentile50
		- percentile75
		
		Key metrics
		- occupancy_rate: number_of_booked_day/number_of_available_day
		- daily_rate: a fee that host charges for 1 night
		- revenue: a monthly revenue. it includes daily rate and cleaning fees
		- cleaning_fee
		- extra_guest_fee: estimated fee per extra guest"
    radiusinmeter: Default is 5,000 meters. We'll include listings within this radius to calculate the income history. Upper limit is 10000 meters. 
        
    """
    url = f"https://airbnb-income-prediction.p.rapidapi.com/getIncomeHistory"
    querystring = {'bedrooms': bedrooms, 'bathrooms': bathrooms, 'countryCode': countrycode, 'coordinate': coordinate, }
    if capacity:
        querystring['capacity'] = capacity
    if radiusinmeter:
        querystring['radiusInMeter'] = radiusinmeter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-income-prediction.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

