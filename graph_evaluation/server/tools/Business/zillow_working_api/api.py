import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def rapidapi_test2(propertyaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is for checking if rapid api is down or not."
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/clientb/byaddress"
    querystring = {'propertyaddress': propertyaddress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clientc_byurl(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "** This endpoint has Neighborhood walk, transit and bike score.
		
		This is a custom endpoint made for a client. Property URL search.
		Input any property url to get results."
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/clientc/byurl"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pricehistory_byzpid(zpid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives you price history of a property. Good for making charts or analysing present value.
		
		**You can get zpid from /by address endpoint under "property info- minimalistic" above."
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/pricehistory/byzpid"
    querystring = {'zpid': zpid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def taxinfo_byzpid(zpid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives you property tax information by zpid. Input any property zpid like: 44466838
		
		If you can't find zpid of a property address, then use the /by property address endpoint to get the zpid from there."
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/taxinfo/byzpid"
    querystring = {'zpid': zpid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_zillow_url(url: str='https://www.zillow.com/homes/3071%20IMPERIAL%20ST%20JACKSONVILLE,%20FL-%2032254/44466838_zpid/', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "put any property url from zillow: 
		ex. https://www.zillow.com/homes/3071%20IMPERIAL%20ST%20JACKSONVILLE,%20FL-%2032254/44466838_zpid/"
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/byurl"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_zpid(zpid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "INPUT: ZPID(44466838)
		Get Property Details By ZPID( you can see the zpid in the zillow url)
		
		If you can't find your zpid, then use /byaddress endpoint above. It works the same."
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/pro/byzpid"
    querystring = {'zpid': zpid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def client_byaddress(propertyaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "** This endpoint has no images URL.
		
		This is a custom endpoint made for a client. Property details by address search.
		Input any property address to get results."
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/client/byaddress"
    querystring = {'propertyaddress': propertyaddress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_zillow_url(url: str='https://www.zillow.com/homes/3071%20IMPERIAL%20ST%20JACKSONVILLE,%20FL-%2032254/44466838_zpid/', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Input Zillow URL: https://www.zillow.com/homes/3071%20IMPERIAL%20ST%20JACKSONVILLE,%20FL-%2032254/44466838_zpid/"
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/pro/byurl"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_property_address(propertyaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "INPUT: Property Address(3071 Imperial St, Jacksonville, FL 32254)
		
		The API will find it's ZPID from property address at backend with 100% accuracy then get's you the property details."
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/pro/byaddress"
    querystring = {'propertyaddress': propertyaddress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_byurl(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by any zillow search URL. You can use any customised filter on Zillow then copy the url and paste it here."
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/search/byurl"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_property_address(propertyaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Minimalistic yet advanced.  The API will find it's ZPID from property address at backend with 100% accuracy then get's you the property details."
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/byaddress"
    querystring = {'propertyaddress': propertyaddress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_byaddress(query: str, page: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Let you search zillow website. you can input area/address or any customised filtered url here. 
		
		If you want to get customised search results use the search/byurl endpoint."
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/search/byaddress"
    querystring = {'query': query, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_zpid(zpid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Property Details By ZPID( you can see the zpid in the zillow url)
		
		If you can't find your zpid, then use /byaddress endpoint above. It works the same."
    
    """
    url = f"https://zillow-working-api.p.rapidapi.com/byzpid"
    querystring = {'zpid': zpid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

