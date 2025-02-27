import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def converts_kilocalories_to_kilojoules(kcal: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "kilocalories to kilojoules"
    
    """
    url = f"https://nutri-score.p.rapidapi.com/v1/convert/kcal/to/kjoule/{kcal}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nutri-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_sodium_g_to_salt_g(gram: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert sodium in gram to salt im gram"
    
    """
    url = f"https://nutri-score.p.rapidapi.com/v1/convert/sodium/to/salt/{gram}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nutri-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_salt_g_to_sodium_g(gram: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert salt in gram to sodium in gram"
    
    """
    url = f"https://nutri-score.p.rapidapi.com/v1/convert/salt/to/sodium/{gram}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nutri-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nutri_score_calculation_of_food(kcal: int, sat_fat_g: int=None, protein_g: int=10, fiber_g: int=None, fruit_veg_percentage: int=None, salt_g: int=None, sugar_g: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Nutri Score for food"
    kcal: Energy in kilocalories
        sat_fat_g: Saturated fat in grams
        protein_g: Protein in grams
        fiber_g: Fiber in grams
        fruit_veg_percentage: Percentage of fruits, vegetables, legumes, nuts as well as rapeseed, walnut and olive oils (0-1)
        salt_g: Salt in grams
        sugar_g: Sugar in grams
        
    """
    url = f"https://nutri-score.p.rapidapi.com/v1/nutri-score/food/{kcal}"
    querystring = {}
    if sat_fat_g:
        querystring['sat_fat_g'] = sat_fat_g
    if protein_g:
        querystring['protein_g'] = protein_g
    if fiber_g:
        querystring['fiber_g'] = fiber_g
    if fruit_veg_percentage:
        querystring['fruit_veg_percentage'] = fruit_veg_percentage
    if salt_g:
        querystring['salt_g'] = salt_g
    if sugar_g:
        querystring['sugar_g'] = sugar_g
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nutri-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_kilojoules_to_kilocalories(kjoule: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert kilojoules to kilocalories"
    
    """
    url = f"https://nutri-score.p.rapidapi.com/v1/convert/kjoule/to/kcal/{kjoule}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nutri-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nutri_score_calculation_of_added_fats(kcal: int, fat_g: int, protein_g: int=12, fruit_veg_percentage: int=None, sugar_g: int=None, salt_g: int=None, sat_fat_g: int=None, fiber_g: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Nutri-Score calculation of added fats (special case)"
    kcal: Energy in kilocalories
        fat_g: Total amount of fat in grams
        protein_g: Protein in grams
        fruit_veg_percentage: Percentage of fruits, vegetables, legumes, nuts as well as rapeseed, walnut and olive oils (0-1)
        sugar_g: Sugar in grams
        salt_g: Salt in grams
        sat_fat_g: Saturated fat in grams
        fiber_g: Fiber in grams
        
    """
    url = f"https://nutri-score.p.rapidapi.com/v1/nutri-score/added_fats/{kcal}"
    querystring = {'fat_g': fat_g, }
    if protein_g:
        querystring['protein_g'] = protein_g
    if fruit_veg_percentage:
        querystring['fruit_veg_percentage'] = fruit_veg_percentage
    if sugar_g:
        querystring['sugar_g'] = sugar_g
    if salt_g:
        querystring['salt_g'] = salt_g
    if sat_fat_g:
        querystring['sat_fat_g'] = sat_fat_g
    if fiber_g:
        querystring['fiber_g'] = fiber_g
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nutri-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nutri_score_calculation_of_beverages(kcal: int, protein_g: int=10, sugar_g: int=None, salt_g: int=None, fiber_g: int=None, sat_fat_g: int=None, fruit_veg_percentage: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Nutri Score of beverages (special case)"
    kcal: Energy in kilocalories
        protein_g: Protein in grams
        sugar_g: Sugar in grams
        salt_g: Salt in grams
        fiber_g: Fiber in grams
        sat_fat_g: Saturated fat in grams
        fruit_veg_percentage: Percentage of fruits, vegetables, legumes, nuts as well as rapeseed, walnut and olive oils (0-1)
        
    """
    url = f"https://nutri-score.p.rapidapi.com/v1/nutri-score/beverages/{kcal}"
    querystring = {}
    if protein_g:
        querystring['protein_g'] = protein_g
    if sugar_g:
        querystring['sugar_g'] = sugar_g
    if salt_g:
        querystring['salt_g'] = salt_g
    if fiber_g:
        querystring['fiber_g'] = fiber_g
    if sat_fat_g:
        querystring['sat_fat_g'] = sat_fat_g
    if fruit_veg_percentage:
        querystring['fruit_veg_percentage'] = fruit_veg_percentage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nutri-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nutri_score_calculation_of_cheese(kcal: int, salt_g: int=None, fruit_veg_percentage: int=None, sat_fat_g: int=None, sugar_g: int=None, protein_g: int=10, fiber_g: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Nutri Score for cheese (special case)"
    kcal: Energy in kilocalories
        salt_g: Salt in grams
        fruit_veg_percentage: Percentage of fruits, vegetables, legumes, nuts as well as rapeseed, walnut and olive oils (0-1)
        sat_fat_g: Saturated fat in grams
        sugar_g: Sugar in grams
        protein_g: Protein in grams
        fiber_g: Fiber in grams
        
    """
    url = f"https://nutri-score.p.rapidapi.com/v1/nutri-score/cheese/{kcal}"
    querystring = {}
    if salt_g:
        querystring['salt_g'] = salt_g
    if fruit_veg_percentage:
        querystring['fruit_veg_percentage'] = fruit_veg_percentage
    if sat_fat_g:
        querystring['sat_fat_g'] = sat_fat_g
    if sugar_g:
        querystring['sugar_g'] = sugar_g
    if protein_g:
        querystring['protein_g'] = protein_g
    if fiber_g:
        querystring['fiber_g'] = fiber_g
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nutri-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

