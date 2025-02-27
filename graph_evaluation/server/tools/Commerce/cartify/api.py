import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getorders(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint retrieves all orders.
		
		Set the authorization to "Bearer admin" to authorize that the admin can check and confirm all orders.
		
		You should get a JSON object with the following attributes;
		id (integer): The unique identifier for the order.
		deliveryDate(Date): The date that the user wants the order to be delivered
		destination(String): Location
		Alongside with the user who ordered the product"
    
    """
    url = f"https://cartify.p.rapidapi.com/users/api/v1/admin/orders"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cartify.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getproducts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gets all products in the database.
		
		A list of products in the store, with the following attributes:
		
		id (integer): The unique identifier for the product.
		title (string): The name of the product.
		category (string): A description of the product.
		price (number): The price of the product.
		image_url (string): A URL to an image of the product."
    
    """
    url = f"https://cartify.p.rapidapi.com/products"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cartify.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getproductbyname(brand: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint filters through the products database and brings out results based on the preferences you want.
		
		For a valid response, it takes a path parameter which is required which takes the name of the product you would want."
    
    """
    url = f"https://cartify.p.rapidapi.com/products/{brand}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cartify.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getproductbycategory(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Do you want to search for a product by category?
		Use the category of the product in the path parameter.
		
		For example,
		Are you looking for Phones? Use Phones in your path parameter"
    
    """
    url = f"https://cartify.p.rapidapi.com/products/p/{category}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cartify.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getordersbyemail(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For a valid response, fill in the email of the user in the path parameter. This is helpful instead of looping from the userId since the users can't have or use the same email. To also verify that it is an authenticated user trying to get his/her order, fill in an authorization header that takes the token of the user.....
		For example, 
		"Bearer <access_token>"
    
    """
    url = f"https://cartify.p.rapidapi.com/users/api/v1/{email}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cartify.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallusers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For a valid response, you would need to have an authorization header and set the role to *Bearer admin*."
    
    """
    url = f"https://cartify.p.rapidapi.com/users"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cartify.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

