import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def betterhealth(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<b>Retrieve up to 12 products with score details and minimal product attributes.</b>
		
		<br /><b><i>You must send us your catalog to receive recommendations within your catalog. </i>Please see <a href="https://docs.greenchoice.co/reference/submitting-your-catalog">Submitting Your Catalog</a> for more details on sending us your catalog.</b>"
    upc: The UPC (GTIN in UPC12 format) for the product
        
    """
    url = f"https://food-diet-nutrition-sustainability.p.rapidapi.com/better/health/{upc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-diet-nutrition-sustainability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def betternutrition(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<b>Retrieve up to 12 products with score details and minimal product attributes.</b>
		
		<br /><b><i>You must send us your catalog to receive recommendations within your catalog. </i>Please see <a href="https://docs.greenchoice.co/reference/submitting-your-catalog">Submitting Your Catalog</a> for more details on sending us your catalog.</b>"
    upc: The UPC (GTIN in UPC12 format) for the product
        
    """
    url = f"https://food-diet-nutrition-sustainability.p.rapidapi.com/better/nutrition/{upc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-diet-nutrition-sustainability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def betteralternatives(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<b>Retrieve up to 12 products with score details and minimal product attributes.</b>
		
		<br /><b><i>You must send us your catalog to receive recommendations within your catalog. </i>Please see <a href="https://docs.greenchoice.co/reference/submitting-your-catalog">Submitting Your Catalog</a> for more details on sending us your catalog.</b>"
    upc: The UPC (GTIN in UPC12 format) for the product
        
    """
    url = f"https://food-diet-nutrition-sustainability.p.rapidapi.com/better/{upc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-diet-nutrition-sustainability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bettersustainable(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<b>Retrieve up to 12 products with score details and minimal product attributes.</b>
		
		<br /><b><i>You must send us your catalog to receive recommendations within your catalog. </i>Please see <a href="https://docs.greenchoice.co/reference/submitting-your-catalog">Submitting Your Catalog</a> for more details on sending us your catalog.</b>"
    upc: The UPC (GTIN in UPC12 format) for the product
        
    """
    url = f"https://food-diet-nutrition-sustainability.p.rapidapi.com/better/sustainable/{upc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-diet-nutrition-sustainability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def betterenvironment(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<b>Retrieve up to 12 products with score details and minimal product attributes.</b>
		
		<br /><b><i>You must send us your catalog to receive recommendations within your catalog. </i>Please see <a href="https://docs.greenchoice.co/reference/submitting-your-catalog">Submitting Your Catalog</a> for more details on sending us your catalog.</b>"
    upc: The UPC (GTIN in UPC12 format) for the product
        
    """
    url = f"https://food-diet-nutrition-sustainability.p.rapidapi.com/better/environment/{upc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-diet-nutrition-sustainability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bettersafety(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<b>Retrieve up to 12 products with score details and minimal product attributes.</b>
		
		<br /><b><i>You must send us your catalog to receive recommendations within your catalog. </i>Please see <a href="https://docs.greenchoice.co/reference/submitting-your-catalog">Submitting Your Catalog</a> for more details on sending us your catalog.</b>"
    upc: The UPC (GTIN in UPC12 format) for the product
        
    """
    url = f"https://food-diet-nutrition-sustainability.p.rapidapi.com/better/safety/{upc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-diet-nutrition-sustainability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def betterprocessing(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<b>Retrieve up to 12 products with score details and minimal product attributes.</b>
		
		<br /><b><i>You must send us your catalog to receive recommendations within your catalog. </i>Please see <a href="https://docs.greenchoice.co/reference/submitting-your-catalog">Submitting Your Catalog</a> for more details on sending us your catalog.</b>"
    upc: The UPC (GTIN in UPC12 format) for the product
        
    """
    url = f"https://food-diet-nutrition-sustainability.p.rapidapi.com/better/processing/{upc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-diet-nutrition-sustainability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def climatelabels(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single product's Carbon Footprint Estimate, Carbon Footpint Value Tag, Carbon Equivalent, Water Footprint Estimate, Water Footprint Value Tag, & Water Equivalent"
    upc: The UPC (GTIN in UPC12 format) for the product
        
    """
    url = f"https://food-diet-nutrition-sustainability.p.rapidapi.com/climate-labels/{upc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-diet-nutrition-sustainability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchenriched(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single product's information, including attributes, insights, and scores."
    upc: The UPC (GTIN in UPC12 format) for the product
        
    """
    url = f"https://food-diet-nutrition-sustainability.p.rapidapi.com/enrich/{upc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-diet-nutrition-sustainability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchdownload(filename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><b>1. Download the updated product file</b><br />Retrieve your product file with new columns "Matched", "Eligible", and "Match Details".
		<br /><b>2. Review the file for data issues</b><br />If the column Matched is "Y" and the column Eligible is "Y", GreenChoice has the data to enrich that product.
		
		<br /><b>4. Request Raw Data Collection</b><br />You may also request GreenChoice to perform Raw Data Collection on any UPCs you wish.<br /><a href="https://bit.ly/meetgkm" target="_blank">Schedule a meeting</a> or <a href="mailto:galen@greenchoice.co">email us</a> for more details. 
		<br /><i>Note: Sending us your reliable data to enable enrichment is faster and more affordable than Raw Data Collection. <br />Please see <a href="https://docs.greenchoice.co/reference/submitting-your-catalog">Submitting Your Catalog</a> for more details on sending us your catalog.
		</i>"
    filename: The name of the file as given in the /match/check output or the email notification
        
    """
    url = f"https://food-diet-nutrition-sustainability.p.rapidapi.com/download/{filename}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-diet-nutrition-sustainability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def relatedproducts(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<b>Retrieve up to 12 products with score details and minimal product attributes.</b>
		
		<br /><b><i>You must send us your catalog to receive recommendations within your catalog. </i>Please see <a href="https://docs.greenchoice.co/reference/submitting-your-catalog">Submitting Your Catalog</a> for more details on sending us your catalog.</b>"
    upc: The UPC (GTIN in UPC12 format) for the product
        
    """
    url = f"https://food-diet-nutrition-sustainability.p.rapidapi.com/related/{upc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-diet-nutrition-sustainability.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

