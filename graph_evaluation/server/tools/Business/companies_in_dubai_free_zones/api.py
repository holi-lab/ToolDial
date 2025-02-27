import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def consultancy_and_bussiness_management_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Consultancy and Bussiness Management** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-consultancy-and-business-management"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def packaging_and_printing_materials_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Packaging and Printing Materials** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-packaging-and-printing-materials"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def engines_and_machines_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Engines and Machines** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-engines-and-machines"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def electronics_electrical_and_ict_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Electronics, Electrical, and ICT** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-electronics-electrical-and-ict"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fz_llc_non_individual_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **FZ-LLC (Non-Individual)** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-fz-llc-non-individual"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fzco_non_individual_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **FZCO (Non-Individual)** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-fzco-non-individual"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fzco_individual_non_individual_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **FZCO (Individual & Non-Individual)** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-fzco-individual-and-non-individual"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fzco_individual_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **FZCO (Individual)** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-fzco-individual"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def branch_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Branch** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-branch"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def machinery_and_equipment_trading_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Machinery and equipment Trading** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-machinery-and-equipment-trading"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def plastic_and_chemical_products_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Plastic and Chemical Products** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-plastic-and-chemical-products"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def engineering_and_building_materials_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Engineering and Building Materials** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-engineering-and-building-materials"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consumer_products_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Consumer Products** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-consumer-products"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pharmaceutical_and_medical_equipments_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Pharmaceutical and Medical Equipments** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-pharmaceutical-and-medical-equipments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def financial_services_insurance_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Financial Services/Insurance** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-financial-services-insurance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coal_oil_and_natural_gas_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Coal, Oil and Natural Gas** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-coal-oil-and-natural-gas"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def food_and_beverages_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Food and Beverages** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-food-and-beverages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def automobiles_and_vehicles_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Automobiles and Vehicles** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-automobiles-and-vehicles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aerospace_aviation_and_related_services_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all **Aerospace, Aviation and related services** active businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-aerospace-aviation-and-related-services"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def logistics_and_freight_active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active **Logistics and Freight** businesses in Free zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-logistics-and-freight"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def active_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns all active businesses in Dubai Free Zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/active-companies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocked_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all blocked business in Free Zones"
    
    """
    url = f"https://companies-in-dubai-free-zones.p.rapidapi.com/blocked-companies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-in-dubai-free-zones.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

