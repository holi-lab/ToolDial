import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def property_rent_estimate(bathrooms: int=2, daysold: int=None, bedrooms: int=4, maxradius: int=None, squarefootage: int=1600, latitude: int=None, longitude: int=None, propertytype: str='Single Family', address: str='5500 Grand Lake Drive, San Antonio, TX, 78244', compcount: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a rent estimate and comparable rental listings for an address or lat/long coordinate. Providing property feature parameters will improve the estimate accuracy. [More info.](https://rapidapi.com/moneals/api/rent-estimate/details)"
    bathrooms: The number of bathrooms in the property. Supports fractions to indicate partial bathrooms
        daysold: The maximum number of days since comparable listings were last seen on the market, with a minimum of 1
        bedrooms: The number of bedrooms in the property
        maxradius: The maximum distance between comparable listings and the subject property, in kilometers. Defaults to 50 if not provided
        squarefootage: The total living area size of the property, in square feet
        latitude: The latitude of the property. The `latitude`/`longitude` can be provided instead of the `address` parameter
        longitude: The longitude of the property. The `latitude`/`longitude` can be provided instead of the `address` parameter
        propertytype: The type of the property. Supported values are: `Single Family`, `Condo`, `Townhouse`, `Manufactured`, `Duplex-Triplex`, `Apartment`
        address: The property address in the format of 'Street, City, State, Zip'. You need to provide either the `address` or the `latitude`/`longitude` parameters
        compcount: The number of comparable listings returned by the API, between 5 and 25. Defaults to 10 if not provided
        
    """
    url = f"https://realtymole-rental-estimate-v1.p.rapidapi.com/rentalPrice"
    querystring = {}
    if bathrooms:
        querystring['bathrooms'] = bathrooms
    if daysold:
        querystring['daysOld'] = daysold
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if maxradius:
        querystring['maxRadius'] = maxradius
    if squarefootage:
        querystring['squareFootage'] = squarefootage
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    if propertytype:
        querystring['propertyType'] = propertytype
    if address:
        querystring['address'] = address
    if compcount:
        querystring['compCount'] = compcount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtymole-rental-estimate-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

