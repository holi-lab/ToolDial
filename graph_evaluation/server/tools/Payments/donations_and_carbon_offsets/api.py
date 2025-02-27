import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v1_donations_show(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the details of a donation you've previously made."
    id: The id of a donation. Ids are returned when a donation is created.
        
    """
    url = f"https://donations-and-carbon-offsets.p.rapidapi.com/api/v1/donations/show"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "donations-and-carbon-offsets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_donations_index(page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a list of donations you've previously made. The donations are returned in order of creation, with the most recent donations appearing first. This endpoint is paginated."
    page: Which page to return. This endpoint is paginated, and returns maximum 30 donations per page.
        
    """
    url = f"https://donations-and-carbon-offsets.p.rapidapi.com/api/v1/donations/index"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "donations-and-carbon-offsets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_nonprofits_show(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves information for a nonprofit."
    id: The id of a nonprofit from the CHANGE network.
        
    """
    url = f"https://donations-and-carbon-offsets.p.rapidapi.com/api/v1/nonprofits/show"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "donations-and-carbon-offsets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_donations_carbon_calculate(weight_lb: int, transportation_method: str='air', distance_mi: int=None, destination_address: int=None, origin_address: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates the donation amount (to CarbonFund 501\(c\)3) needed to offset a physical shipment. This calculation depends on the weight, primary transportation method, and distance of the shipment. Provide the distance of the shipment using the origin and destination address, or directly with the number of miles. For convenience, this endpoint also returns the id of the nonprofit CarbonFund, for making a subsequent donation to. See the [Carbon offsets guide](/recipes/carbon-offsets/) for more on using this endpoint."
    weight_lb: The total weight (in pounds) of the shipment.
        transportation_method: The primary transportation method of the shipment.
        distance_mi: The total distance (in miles) of the shipment. You can use this parameter in place of `origin_address` and `destination_address`.
        destination_address: The destination zip code (US only) of the shipment. If you send this parameter, also send `origin_address`.
        origin_address: The origin zip code (US only) of the shipment. If you send this parameter, also send `destination_address`.
        
    """
    url = f"https://donations-and-carbon-offsets.p.rapidapi.com/api/v1/donations/carbon_calculate"
    querystring = {'weight_lb': weight_lb, }
    if transportation_method:
        querystring['transportation_method'] = transportation_method
    if distance_mi:
        querystring['distance_mi'] = distance_mi
    if destination_address:
        querystring['destination_address'] = destination_address
    if origin_address:
        querystring['origin_address'] = origin_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "donations-and-carbon-offsets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_donations_carbon_stats(is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Measures your carbon offset impact in relatable terms. Provide the id of a donation to CarbonFund to see stats about that specific donation. If you omit the donation id, this endpoint returns aggregate stats for all of your CarbonFund donations."
    is_id: The id of a donation to the CarbonFund nonprofit. Ids are returned when a donation is created. If an ID is not provided, the total stats for all donations to CarbonFund are returned.
        
    """
    url = f"https://donations-and-carbon-offsets.p.rapidapi.com/api/v1/donations/carbon_stats"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "donations-and-carbon-offsets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_donations_crypto_calculate(currency: str, count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates the donation amount (to CarbonFund 501\(c\)3) needed to offset a cryptocurrency transaction. For convenience, this endpoint also returns the id of the nonprofit CarbonFund, for making a subsequent donation to. See the [Carbon offsets guide](/recipes/carbon-offsets/) for more on using this endpoint."
    currency: The currency of the transaction.
        count: The number of transactions to offset.
        
    """
    url = f"https://donations-and-carbon-offsets.p.rapidapi.com/api/v1/donations/crypto_calculate"
    querystring = {'currency': currency, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "donations-and-carbon-offsets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_nonprofits_list(name: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a list of nonprofits whose names match the provided name. This endpoint is paginated."
    name: A string to search.
        page: The page to return. This endpoint is paginated, and returns up to 30 nonprofits at a time.
        
    """
    url = f"https://donations-and-carbon-offsets.p.rapidapi.com/api/v1/nonprofits/list"
    querystring = {}
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "donations-and-carbon-offsets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

