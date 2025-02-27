import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def blocks(block_id: str='minecraft:stone', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve blocks information:
		
		- **block_id** such as "minecraft:stone"
		- **block_group** such as "Natural"
		- **light_emission** such as 15 for torches or 0 for non-emitting blocks (MC unit)
		- **burn_chance** to indicate the probability of the block catching fire 
		- **flammable** is true if the block can catch fire otherwise is false
		- **hardness** to indicate how hard a block is (MC unit)
		- **blast_resistance** to indicate how strong is the block against explosions
		- **friction** is a float value that represents block friction
		- **speed_factor** is a float value that indicates whether  the block slows you
		- **jump_factor**  is a float value that indicates whether  the block should make you jump higher
		- **rarity** such as "COMMON"
		- **tool_required** is true if the block requires the use of a tool to generate a drop
		- **tags** represents the list of tags attached to that block"
    
    """
    url = f"https://mcapi4.p.rapidapi.com/api/blocks"
    querystring = {}
    if block_id:
        querystring['block_id'] = block_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mcapi4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def items(item_id: str='minecraft:coal', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve items information:
		
		- **item_id** is the id of the item like "minecraft:stick"
		- **item_group** is the group of the item like "Ingredients"
		- **stack_size** is the max amount of items in one stack
		- **fire_resistant** is true if the item doesn't burn
		- **rarity** is the item rarity such as "COMMON"
		- **is_food** is true if the item can be eaten
		- **is_damageable** is true if the item can be damaged when used
		- **is_enchantable** is true if the item can be enchanted
		- **durability** indicates how many times the item can be used
		- **tags** are the tags applied to this item"
    
    """
    url = f"https://mcapi4.p.rapidapi.com/api/items"
    querystring = {}
    if item_id:
        querystring['item_id'] = item_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mcapi4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def images(name: str, width: str='800', height: str='800', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve item or block image (to be removed):
		
		- **name** is the name of the block or item (without minecraft;)
		- **width** is the desired width of the returned image
		- **height** is the desired height of the returned image"
    
    """
    url = f"https://mcapi4.p.rapidapi.com/api/images"
    querystring = {'name': name, }
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mcapi4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tags(father_tag: str='minecraft:logs_that_burn', tag_id: str='minecraft:acacia_logs', name: str='acacia_logs', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve blocks and items tags"
    
    """
    url = f"https://mcapi4.p.rapidapi.com/api/tags"
    querystring = {}
    if father_tag:
        querystring['father_tag'] = father_tag
    if tag_id:
        querystring['tag_id'] = tag_id
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mcapi4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def total_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get total count of entries"
    
    """
    url = f"https://mcapi4.p.rapidapi.com/api/totalcount"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mcapi4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recipes(recipe_type: str='minecraft:crafting_shaped', output: str='minecraft:stone', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve recipes"
    
    """
    url = f"https://mcapi4.p.rapidapi.com/api/recipes"
    querystring = {}
    if recipe_type:
        querystring['recipe_type'] = recipe_type
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mcapi4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advancements(category: str='adventure', advancement_id: str='adventure.bullseye', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve advancements:
		
		- **advancement_id** is the ID made of the category+title
		 - **category** is the category of the advancement (adventure, husbandry, nether, story or end)
		 - **title** is the in-game title
		 - **description** is a short description of the advancement
		  - **frame** is the type of background used
		  - **icon** is the icon inside of the frame (block or item without minecraft:)
		  - **parent_advancement** is the ID of the previous adv. to be achieved"
    
    """
    url = f"https://mcapi4.p.rapidapi.com/api/advancements"
    querystring = {}
    if category:
        querystring['category'] = category
    if advancement_id:
        querystring['advancement_id'] = advancement_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mcapi4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

