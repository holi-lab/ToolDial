import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def main_fetch_endpoint(equipment: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to access all armor and weapons
		Format of the armor fetch is {rank}-{amorType}
		Where rank is either "lr" or "hr" and armorType is one of the following:
		Head
		Chest
		Arms
		Waist
		Legs
		
		Example: "lr-waist" to get all low-rank waist armor.
		
		To get a type of weapon is a selection of these slugs:
		gs - Greatswords
		ls - Longswords
		sns - Swords and Shields
		db - Dual Blades
		h - Hammers
		hh - Hunting Horns
		l - Lances
		gl - Gunlanes
		sa - Switch Axes
		cb - Charge Blades
		ig - Insect Glaives
		lbg - Low Bow Guns
		hbg - Heavy Bow Guns
		b - Bows"
    
    """
    url = f"https://monster-hunter-rise-equipment-database.p.rapidapi.com/{equipment}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monster-hunter-rise-equipment-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

