import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_mini_world_changes(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint will help you get sigle World Changes in our data base by World"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/worldboard"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_imbuements_for_epiphany_magic_skillboost(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint will get all imbuement for type Epiphany
		
		Epiphany (Magic Skillboost)
		Elvish Talisman 
		Broken Shamanic Staff 
		Strand of Medusa Hair"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/imbuements/epiphany"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_imbuements_for_quara_scale_ice_protection(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint will get all imbuement for type Quara Scale
		
		Quara Scale (Ice Protection)
		Winter Wolf Fur 
		Thick Fur 
		Deepling Warts"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/imbuements/quara_scale"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_imbuements_for_cloud_fabric_energy_protection(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint will get all imbuement for type Cloud Fabric
		
		Cloud Fabric (Energy Protection)
		Wyvern Talisman 
		Crawler Head Plating 
		Wyrm Scale"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/imbuements/cloud_fabric"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_imbuements_for_demon_presence_holy_protection(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint will get all imbuement for type Demon Presence
		
		Demon Presence (Holy Protection)
		Cultish Robe 
		Cultish Mask 
		Hellspawn Tail"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/imbuements/demon_presence"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_imbuements_for_precision_distance_skillboost(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint will get all imbuement for type Precision
		
		Precision (Distance Skillboost)
		Elven Scouting Glass 
		Elven Hoof 
		Metal Spike"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/imbuements/precision"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_imbuements_for_snake_skin_earth_protection(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint will get all imbuement for type Snake skin
		
		Snake Skin (Earth Protection)
		Brimstone Fangs 
		Piece of Swampling Wood 
		Snake Skin"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/imbuements/snake_skin"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_imbuements_for_strike_critical_damage(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint will get all imbuement for type Strike
		
		Strike (Critical Damage)
		Protective Charm 
		Sabretooth 
		Vexclaw Talon"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/imbuements/strike"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_imbuements_for_vampirism_life_steal(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint will get all imbuement for type Vampirism
		
		Vampirism (Life steal)
		Vampire Teeth 
		Bloody Pincers 
		Piece of Dead Brain"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/imbuements/vampirism"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_imbuements_items(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint will get all imbuement"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/imbuements"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_history_of_item(date: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In this enpoint we can deliver the history of item.
		you will need to request in this pattern.
		
		Key: NameItem::World::TypeItem
		Key: Gold Token::Wizera::Valuable
		
		Date:2022-01-01
		Date: yyyy-MM-dd
		
		Date: 2022-01
		Date: yyyy-MM"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/history"
    querystring = {'Date': date, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_single_item(world: str, nameitem: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint will help you get sigle item in our data base by World"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/single"
    querystring = {'World': world, 'NameItem': nameitem, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_valuable_items(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get all Valuble item in our data base."
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/valuable"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_imbuements_for_lich_shroud_death_protection(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint will get all imbuement for type Lich Shroud
		
		Lich Shroud (Death Protection)
		Flask of Embalming Fluid 
		Gloom Wolf Fur 
		Mystical Hourglass"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/imbuements/lich_shroud"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_imbuements_for_dragon_hide_fire_protection(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint will get all imbuement for type Dragon Hide
		
		Dragon Hide (Fire Protection)
		Blazing Bone 
		Green Dragon Leather 
		Draken Sulphur"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/imbuements/dragon_hide"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_imbuements_for_void_mana_steal(world: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint will get all imbuement for type Void
		
		Void (Mana Steal)
		Rope Belt 
		Silencer Claws 
		Some Grimeleech Wings"
    
    """
    url = f"https://tibia-items.p.rapidapi.com/tibia-item/imbuements/void"
    querystring = {'World': world, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tibia-items.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

