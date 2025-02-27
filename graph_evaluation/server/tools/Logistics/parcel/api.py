import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_a_return_portal_url(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lets you see which of your branded Return portals is associated with a specific parcel based on the provided parcel `id`.
		The URL which is retrieved will link directly to the parcel in the Sendcloud Return portal, so a **return parcel** can be created immediately based on the outgoing shipment.
		If no Return portal is configured, or if no brand is connected to the parcel, this endpoint will return a status code 404 error."
    id: ID of the parcel
        
    """
    url = f"https://parcel3.p.rapidapi.com/parcels/{is_id}/return_portal_url"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parcel3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_list_of_parcels(parcel_status: int=None, errors: str='verbose-carrier', tracking_number: str=None, cursor: str=None, ids: str=None, order_number: str=None, announced_after: str='2018-02-26T11:01:47.505309+00:00', updated_after: str='2018-02-26T11:01:47.505309+00:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of all the parcels which you have created or imported into your Sendcloud account under your API credentials. You can filter the results based on the query parameters provided below, in order to retrieve a specific parcel or list of parcels which match the defined criteria."
    parcel_status: Returns parcels that have the requested status. For a list of possible statuses, see the [Parcel statuses](openapi.yaml/paths/~1parcels~1statuses) endpoint
        errors: If you are using this API for development purposes, you might want to use the errors query string in the URL. This allows you to visualize errors from the carrier.
        tracking_number: Returns parcels that match a specified tracking number
        cursor: Next and previous token that is used to paginate. The token is included in the response.
        ids: Filter results using a list of Parcel IDs. This is a comma separated list of IDs, it may not contain more then 100 IDs.
        order_number: Returns an order that matches a specific `order_number` property from your parcels
        announced_after: Returns all orders which have been updated in our system after you given time. You can use the value of ISO 8601 DateTime string like this
        updated_after: Returns all orders which have been updated in our system after a given time. You can use the value of ISO 8601 DateTime string like this
        
    """
    url = f"https://parcel3.p.rapidapi.com/parcels"
    querystring = {}
    if parcel_status:
        querystring['parcel_status'] = parcel_status
    if errors:
        querystring['errors'] = errors
    if tracking_number:
        querystring['tracking_number'] = tracking_number
    if cursor:
        querystring['cursor'] = cursor
    if ids:
        querystring['ids'] = ids
    if order_number:
        querystring['order_number'] = order_number
    if announced_after:
        querystring['announced_after'] = announced_after
    if updated_after:
        querystring['updated_after'] = updated_after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parcel3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_parcel(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a specific parcel created under your Sendcloud credentials, based on the parcel `id`."
    id: The id of the parcel you want to retrieve
        
    """
    url = f"https://parcel3.p.rapidapi.com/parcels/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parcel3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

