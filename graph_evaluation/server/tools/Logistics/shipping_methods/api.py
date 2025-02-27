import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_a_shipping_method(is_id: int, sender_address: str=None, is_return: bool=None, service_point_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return information about a shipping method based on the provided shipping method `id` and your **default sender address**. 
		
		As described in <a href="List-of-all-shipping-methods">**List all shipping methods**</a>, to retrieve information about a shipping method which operates in a different country than your default sender address, provide a different `sender_address` `id`.
		
		**Example:**
		
		<span style="background-color: #1D97FF"> GET</span> `https://panel.sendcloud.sc/api/v2/shipping_methods/365?sender_address=102964`
		
		
		```json
		{
			"shipping_method": {
				"id": 365,
				"name": "Colissimo Service Point 0.75-1kg",
				"carrier": "colissimo",
				"min_weight": "0.751",
				"max_weight": "1.001",
				"service_point_input": "required",
				"price": 0,
				"countries": [
		  ```"
    id: Shipping method id
        sender_address: The ID of the sender address for which you would like to know if the given shipping method is available.
        is_return: If set to `true` the endpoint will return the  shipping method only if  it is a return shipping method.
        service_point_id: The ID of the service point for which you would like to know if the given shipping method is available.
        
    """
    url = f"https://shipping-methods.p.rapidapi.com/shipping_methods/{is_id}"
    querystring = {}
    if sender_address:
        querystring['sender_address'] = sender_address
    if is_return:
        querystring['is_return'] = is_return
    if service_point_id:
        querystring['service_point_id'] = service_point_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shipping-methods.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_list_of_shipping_methods(is_return: bool=None, sender_address: str=None, service_point_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return a detailed list of all the shipping methods which are available to you under your Sendcloud credentials. You can use this endpoint to find a specific shipping method `id`, which you can then specify in your request to [Create a parcel](paths/~1parcels/post). 
		
		If a shipping method `id` value is present, and if the `request_label` parameter has the value `true`, then a shipping label is created and the parcel is announced. 
		
		The shipping methods returned are based on the following factors:
		1. The carriers you have <a href="Getting-started">**enabled**</a> in your Sendcloud account; 
		2. (Optional) The direct <a href="Contracts">**carrier contracts**</a> you have connected; and,
		3. Your <a href="sender-addresses">**sender address**</a>
		
		<!-- theme: info -->
		
		>**Tip:** Via this endpoint you can only retrieve shipping methods based on three parameters: `sender_address`, `service_point_id` and `is_return`. If you need to filter the results because yourequire a method which contains a specific shipping functionality or other criteria, you can refer to the <a href="Get-shipping-products">**Get shipping products**</a> endpoint. 
		
		---
		
		### Specifying a sender address
		
		You can have multiple sender addresses stored in your Sendcloud account. This endpoint will return the shipping methods associated with your **default** sender address, **unless** you provide a specific `sender_adress` `id`. 
		
		<!-- theme: info -->
		
		> **Tip:** You can find the `id` for each of your sender addresses via the <a href="get-a-single-sender-address">**Get a single sender address**</a> endpoint.
		
		### Use case
		
		For example, your default sender address may be based in the Netherlands, but you have a second sender address based in Austria. If you don't specify a `sender_address` `id`, this endpoint will **only** return shipping methods applicable for shipments from the Netherlands. 
		
		To see shipping methods applicable for Austria, e.g. from DPD Austria, specify the `id` for your Austrian sender address in the HTTP request. The retrieved results will now include carriers such as Post AT, DPD Austria, etc, depending on your <a href="enabling-carriers">enabled carriers</a>). 
		
		**Example:** 
		
		<span style="background-color: #1D97FF">GET</span> `https://panel.sendcloud.sc/api/v2/shipping_methods?sender_address={ID}`
		
		
		---
		
		### Use cases
		
		- **Find a service point delivery shipping method**
		
		If you want to retrieve a list of shipping methods which are applicable for **service point delivery**, provide a `service_point_id` in the request. You can find an appropriate `service_point_id` via the <a href="Service-points-api">**Service Points API**</a>. 
		
		
		- **Find a suitable shipping method for a return**
		
		Return shipping methods are treated differently than methods for outgoing parcels. If you want to filter the results to show only the methods which you can apply to return parcels, include the argument `is_return: "true"`. 
		
		- **Invalid shipment ID Error message**
		If you try to <a ref="Create-a-parcel">**Create a parcel**</a> but receive the error message "Invalid shipment id", this could be because the specified `id` relates to a shipping method which is not possible for the given destination address. 
		
		For example, if you need to ship a parcel internationally but the specific shipping method only supports national (domestic) shipping, then you would need to lookup a new shipping method `id` which supports method of delivery and change the request."
    is_return: If set to `true` the endpoint will return shipping methods which can be used  for making a return  shipment.
        sender_address: The ID of the sender address for which you would like to know the available shipping methods. If you want to retrieve all available shipping methods, please use `all` as a value for this parameter.
        service_point_id: The ID of the service point for which you would like to know the available shipping methods.
        
    """
    url = f"https://shipping-methods.p.rapidapi.com/shipping_methods"
    querystring = {}
    if is_return:
        querystring['is_return'] = is_return
    if sender_address:
        querystring['sender_address'] = sender_address
    if service_point_id:
        querystring['service_point_id'] = service_point_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shipping-methods.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

