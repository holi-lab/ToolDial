import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def customers_customerid_orders(customerid: str, productid: str=None, countrycode: str=None, connectiontype: str=None, status: str=None, requestid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Returns a list of all your orders.</p><p>Use this to see all orders you have ever placed with the system.  Orders in all states will be shown, including canceled orders.</p><p>This command allows you to search the list of orders by various filter parameters.</p>"
    customerid: The id of the customer object of your account
        productid: Optionally filter order results by the product type. Acceptable values list
        countrycode: An ISO Alpha2 or ISO Alpha3 country code by which to filter results
        connectiontype: Optionally filter order results by the connection type in use. Acceptable values list = [pstn, direct]
        status: Optionally filter order results by the status; accepted values 
        requestid: Any value meaningful to you that correlated orders together
        
    """
    url = f"https://as6.p.rapidapi.com/customers/{customerid}/orders"
    querystring = {}
    if productid:
        querystring['productId'] = productid
    if countrycode:
        querystring['countryCode'] = countrycode
    if connectiontype:
        querystring['connectionType'] = connectiontype
    if status:
        querystring['status'] = status
    if requestid:
        querystring['requestId'] = requestid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "as6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def customers_customerid_orders_orderid(orderid: str, customerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Returns a specific order, identified by its id.</p><p>Use this to see a specific order you have placed with the system.</p>"
    orderid: The id of the order
        customerid: The id of the customer object of your account
        
    """
    url = f"https://as6.p.rapidapi.com/customers/{customerid}/orders/{orderid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "as6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

