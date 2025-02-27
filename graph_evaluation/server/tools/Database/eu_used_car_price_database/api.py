import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def used_car_price(make: str, model: str, yearto: str='2019', yearfrom: str='2017', mileageto: str='150000', mileagefrom: str='100000', fuel: str='Diesel', gearbox: str='Automatic', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Analyze EU used car market prices."
    fuel: diesel, benzine
        gearbox: Manual, Automatic
        
    """
    url = f"https://eu-used-car-price-database.p.rapidapi.com/"
    querystring = {'make': make, 'model': model, }
    if yearto:
        querystring['yearTo'] = yearto
    if yearfrom:
        querystring['yearFrom'] = yearfrom
    if mileageto:
        querystring['mileageTo'] = mileageto
    if mileagefrom:
        querystring['mileageFrom'] = mileagefrom
    if fuel:
        querystring['fuel'] = fuel
    if gearbox:
        querystring['gearbox'] = gearbox
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eu-used-car-price-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

