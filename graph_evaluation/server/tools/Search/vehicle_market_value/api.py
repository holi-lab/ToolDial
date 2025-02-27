import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vehicle_market_value_by_license_plate(state_code: str, license_plate: str, mileage: str=None, period: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vehicle Market Value by License Plate"
    state_code: State Code
AL,AK,AZ,AR,CA,CO,CT,DE,DC,FL,GA,HI,ID,IL,IN,IA,KS,KY,LA,ME,MD,MA,MI,MN,MS,MO,MT,NE,NV,NH,NJ,NM,NY,NC,ND,OH,OK,OR,PA,RI,SC,SD,TN,TX,UT,VT,VA,WA,WV,WI,WY
        license_plate: License plate number
        mileage: The current mileage of the vehicle, or if not specified, the average mileage is used.For example: 50000

        period: Number of days to look back for sales data ,maximum value: 365.For example: 180
        
    """
    url = f"https://vehicle-market-value.p.rapidapi.com/vmv"
    querystring = {'state_code': state_code, 'license_plate': license_plate, }
    if mileage:
        querystring['mileage'] = mileage
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vehicle-market-value.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vehicle_market_value_by_vin(vin: str, mileage: str=None, period: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vehicle Market Value by VIN"
    vin: The vehicle identification number
        mileage: The current mileage of the vehicle, or if not specified, the average mileage is used.For example: 50000
        period: Number of days to look back for sales data ,maximum value: 365.For example: 180
        
    """
    url = f"https://vehicle-market-value.p.rapidapi.com/vmv"
    querystring = {'vin': vin, }
    if mileage:
        querystring['mileage'] = mileage
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vehicle-market-value.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

