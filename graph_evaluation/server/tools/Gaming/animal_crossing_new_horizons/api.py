import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_dress_up_api_v1_dress_up_get(mannequin_piece: str=None, secondary_shape: str=None, page_size: int=50, seasonal_availability: str=None, name_neq: str=None, version: str=None, internal_id: str=None, catalog: str=None, sell: str=None, page: int=1, primary_shape: str=None, name: str=None, source_notes: str=None, filename: str=None, villager_equippable: str=None, color_1: str=None, name_ilike: str=None, source: str=None, unique_entry_id: str=None, color_2: str=None, buy: str=None, style: str=None, label_themes: str=None, diy: str=None, variation: str=None, name_like: str=None, size: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/dress_up"
    querystring = {}
    if mannequin_piece:
        querystring['mannequin_piece'] = mannequin_piece
    if secondary_shape:
        querystring['secondary_shape'] = secondary_shape
    if page_size:
        querystring['page_size'] = page_size
    if seasonal_availability:
        querystring['seasonal_availability'] = seasonal_availability
    if name_neq:
        querystring['name__neq'] = name_neq
    if version:
        querystring['version'] = version
    if internal_id:
        querystring['internal_id'] = internal_id
    if catalog:
        querystring['catalog'] = catalog
    if sell:
        querystring['sell'] = sell
    if page:
        querystring['page'] = page
    if primary_shape:
        querystring['primary_shape'] = primary_shape
    if name:
        querystring['name'] = name
    if source_notes:
        querystring['source_notes'] = source_notes
    if filename:
        querystring['filename'] = filename
    if villager_equippable:
        querystring['villager_equippable'] = villager_equippable
    if color_1:
        querystring['color_1'] = color_1
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if source:
        querystring['source'] = source
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if color_2:
        querystring['color_2'] = color_2
    if buy:
        querystring['buy'] = buy
    if style:
        querystring['style'] = style
    if label_themes:
        querystring['label_themes'] = label_themes
    if diy:
        querystring['diy'] = diy
    if variation:
        querystring['variation'] = variation
    if name_like:
        querystring['name__like'] = name_like
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_wall_mounted_api_v1_wall_mounted_get(hha_series: str=None, color_2: str=None, hha_concept_2: str=None, lighting_type: str=None, size: str=None, source: str=None, internal_id: str=None, hha_concept_1: str=None, buy: str=None, kit_cost: str=None, version: str=None, door_deco: str=None, filename: str=None, body_customize: str=None, source_notes: str=None, name: str=None, name_like: str=None, catalog: str=None, variant_id: str=None, page_size: int=50, interact: str=None, color_1: str=None, outdoor: str=None, hha_set: str=None, tag: str=None, sell: str=None, pattern_customize: str=None, pattern_title: str=None, diy: str=None, pattern: str=None, unique_entry_id: str=None, body_title: str=None, page: int=1, name_ilike: str=None, name_neq: str=None, variation: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/wall_mounted"
    querystring = {}
    if hha_series:
        querystring['hha_series'] = hha_series
    if color_2:
        querystring['color_2'] = color_2
    if hha_concept_2:
        querystring['hha_concept_2'] = hha_concept_2
    if lighting_type:
        querystring['lighting_type'] = lighting_type
    if size:
        querystring['size'] = size
    if source:
        querystring['source'] = source
    if internal_id:
        querystring['internal_id'] = internal_id
    if hha_concept_1:
        querystring['hha_concept_1'] = hha_concept_1
    if buy:
        querystring['buy'] = buy
    if kit_cost:
        querystring['kit_cost'] = kit_cost
    if version:
        querystring['version'] = version
    if door_deco:
        querystring['door_deco'] = door_deco
    if filename:
        querystring['filename'] = filename
    if body_customize:
        querystring['body_customize'] = body_customize
    if source_notes:
        querystring['source_notes'] = source_notes
    if name:
        querystring['name'] = name
    if name_like:
        querystring['name__like'] = name_like
    if catalog:
        querystring['catalog'] = catalog
    if variant_id:
        querystring['variant_id'] = variant_id
    if page_size:
        querystring['page_size'] = page_size
    if interact:
        querystring['interact'] = interact
    if color_1:
        querystring['color_1'] = color_1
    if outdoor:
        querystring['outdoor'] = outdoor
    if hha_set:
        querystring['hha_set'] = hha_set
    if tag:
        querystring['tag'] = tag
    if sell:
        querystring['sell'] = sell
    if pattern_customize:
        querystring['pattern_customize'] = pattern_customize
    if pattern_title:
        querystring['pattern_title'] = pattern_title
    if diy:
        querystring['diy'] = diy
    if pattern:
        querystring['pattern'] = pattern
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if body_title:
        querystring['body_title'] = body_title
    if page:
        querystring['page'] = page
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if name_neq:
        querystring['name__neq'] = name_neq
    if variation:
        querystring['variation'] = variation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tools_api_v1_tools_get(diy: str=None, name: str=None, page_size: int=50, name_ilike: str=None, source: str=None, buy: str=None, source_notes: str=None, filename: str=None, set: str=None, unique_entry_id: str=None, sell: str=None, miles_price: str=None, variant_id: str=None, body_title: str=None, page: int=1, color_1: str=None, internal_id: str=None, uses: str=None, stack_size: str=None, name_like: str=None, variation: str=None, version: str=None, customize: str=None, color_2: str=None, name_neq: str=None, kit_cost: str=None, size: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/tools"
    querystring = {}
    if diy:
        querystring['diy'] = diy
    if name:
        querystring['name'] = name
    if page_size:
        querystring['page_size'] = page_size
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if source:
        querystring['source'] = source
    if buy:
        querystring['buy'] = buy
    if source_notes:
        querystring['source_notes'] = source_notes
    if filename:
        querystring['filename'] = filename
    if set:
        querystring['set'] = set
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if sell:
        querystring['sell'] = sell
    if miles_price:
        querystring['miles_price'] = miles_price
    if variant_id:
        querystring['variant_id'] = variant_id
    if body_title:
        querystring['body_title'] = body_title
    if page:
        querystring['page'] = page
    if color_1:
        querystring['color_1'] = color_1
    if internal_id:
        querystring['internal_id'] = internal_id
    if uses:
        querystring['uses'] = uses
    if stack_size:
        querystring['stack_size'] = stack_size
    if name_like:
        querystring['name__like'] = name_like
    if variation:
        querystring['variation'] = variation
    if version:
        querystring['version'] = version
    if customize:
        querystring['customize'] = customize
    if color_2:
        querystring['color_2'] = color_2
    if name_neq:
        querystring['name__neq'] = name_neq
    if kit_cost:
        querystring['kit_cost'] = kit_cost
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_socks_api_v1_socks_get(miles_price: str=None, color_1: str=None, buy: str=None, variation: str=None, name_ilike: str=None, name_like: str=None, diy: str=None, version: str=None, name: str=None, name_neq: str=None, internal_id: str=None, unique_entry_id: str=None, page: int=1, page_size: int=50, villager_equippable: str=None, style: str=None, source_notes: str=None, sell: str=None, filename: str=None, catalog: str=None, seasonal_availability: str=None, mannequin_piece: str=None, size: str=None, label_themes: str=None, color_2: str=None, source: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/socks"
    querystring = {}
    if miles_price:
        querystring['miles_price'] = miles_price
    if color_1:
        querystring['color_1'] = color_1
    if buy:
        querystring['buy'] = buy
    if variation:
        querystring['variation'] = variation
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if name_like:
        querystring['name__like'] = name_like
    if diy:
        querystring['diy'] = diy
    if version:
        querystring['version'] = version
    if name:
        querystring['name'] = name
    if name_neq:
        querystring['name__neq'] = name_neq
    if internal_id:
        querystring['internal_id'] = internal_id
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    if villager_equippable:
        querystring['villager_equippable'] = villager_equippable
    if style:
        querystring['style'] = style
    if source_notes:
        querystring['source_notes'] = source_notes
    if sell:
        querystring['sell'] = sell
    if filename:
        querystring['filename'] = filename
    if catalog:
        querystring['catalog'] = catalog
    if seasonal_availability:
        querystring['seasonal_availability'] = seasonal_availability
    if mannequin_piece:
        querystring['mannequin_piece'] = mannequin_piece
    if size:
        querystring['size'] = size
    if label_themes:
        querystring['label_themes'] = label_themes
    if color_2:
        querystring['color_2'] = color_2
    if source:
        querystring['source'] = source
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reactions_api_v1_reactions_get(name_ilike: str=None, internal_id: str=None, page: int=1, page_size: int=50, source_notes: str=None, unique_entry_id: str=None, name_neq: str=None, source: str=None, name_like: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/reactions"
    querystring = {}
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if internal_id:
        querystring['internal_id'] = internal_id
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    if source_notes:
        querystring['source_notes'] = source_notes
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if name_neq:
        querystring['name__neq'] = name_neq
    if source:
        querystring['source'] = source
    if name_like:
        querystring['name__like'] = name_like
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fossils_api_v1_fossils_get(page_size: int=50, unique_entry_id: str=None, page: int=1, internal_id: str=None, museum: str=None, size: str=None, filename: str=None, interact: str=None, catalog: str=None, buy: str=None, color_2: str=None, name_ilike: str=None, name_neq: str=None, source: str=None, version: str=None, name_like: str=None, color_1: str=None, sell: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/fossils"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if page:
        querystring['page'] = page
    if internal_id:
        querystring['internal_id'] = internal_id
    if museum:
        querystring['museum'] = museum
    if size:
        querystring['size'] = size
    if filename:
        querystring['filename'] = filename
    if interact:
        querystring['interact'] = interact
    if catalog:
        querystring['catalog'] = catalog
    if buy:
        querystring['buy'] = buy
    if color_2:
        querystring['color_2'] = color_2
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if name_neq:
        querystring['name__neq'] = name_neq
    if source:
        querystring['source'] = source
    if version:
        querystring['version'] = version
    if name_like:
        querystring['name__like'] = name_like
    if color_1:
        querystring['color_1'] = color_1
    if sell:
        querystring['sell'] = sell
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_achievements_api_v1_achievements_get(num_of_tiers: str=None, reward_tier_3: str=None, page: int=1, reward_tier_5: str=None, version: str=None, tier_5: str=None, tier_4: str=None, reward_tier_1: str=None, award_criteria_like: str=None, page_size: int=50, reward_tier_4: str=None, tier_1: str=None, award_criteria: str=None, name_ilike: str=None, unique_entry_id: str=None, internal_id: str=None, reward_tier_2: str=None, name_like: str=None, sequential: str=None, tier_2: str=None, tier_3: str=None, reward_tier_6: str=None, name: str=None, internal_name: str=None, internal_category: str=None, name_neq: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/achievements"
    querystring = {}
    if num_of_tiers:
        querystring['num_of_tiers'] = num_of_tiers
    if reward_tier_3:
        querystring['reward_tier_3'] = reward_tier_3
    if page:
        querystring['page'] = page
    if reward_tier_5:
        querystring['reward_tier_5'] = reward_tier_5
    if version:
        querystring['version'] = version
    if tier_5:
        querystring['tier_5'] = tier_5
    if tier_4:
        querystring['tier_4'] = tier_4
    if reward_tier_1:
        querystring['reward_tier_1'] = reward_tier_1
    if award_criteria_like:
        querystring['award_criteria__like'] = award_criteria_like
    if page_size:
        querystring['page_size'] = page_size
    if reward_tier_4:
        querystring['reward_tier_4'] = reward_tier_4
    if tier_1:
        querystring['tier_1'] = tier_1
    if award_criteria:
        querystring['award_criteria'] = award_criteria
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if internal_id:
        querystring['internal_id'] = internal_id
    if reward_tier_2:
        querystring['reward_tier_2'] = reward_tier_2
    if name_like:
        querystring['name__like'] = name_like
    if sequential:
        querystring['sequential'] = sequential
    if tier_2:
        querystring['tier_2'] = tier_2
    if tier_3:
        querystring['tier_3'] = tier_3
    if reward_tier_6:
        querystring['reward_tier_6'] = reward_tier_6
    if name:
        querystring['name'] = name
    if internal_name:
        querystring['internal_name'] = internal_name
    if internal_category:
        querystring['internal_category'] = internal_category
    if name_neq:
        querystring['name__neq'] = name_neq
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_headwear_api_v1_headwear_get(size: str=None, name_ilike: str=None, catalog: str=None, unique_entry_id: str=None, name_neq: str=None, label_themes: str=None, type: str=None, page_size: int=50, buy: str=None, page: int=1, filename: str=None, source: str=None, name: str=None, mannequin_piece: str=None, variation: str=None, style: str=None, miles_price: str=None, color_1: str=None, internal_id: str=None, name_like: str=None, source_notes: str=None, villager_equippable: str=None, version: str=None, seasonal_availability: str=None, sell: str=None, color_2: str=None, diy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/headwear"
    querystring = {}
    if size:
        querystring['size'] = size
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if catalog:
        querystring['catalog'] = catalog
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if name_neq:
        querystring['name__neq'] = name_neq
    if label_themes:
        querystring['label_themes'] = label_themes
    if type:
        querystring['type'] = type
    if page_size:
        querystring['page_size'] = page_size
    if buy:
        querystring['buy'] = buy
    if page:
        querystring['page'] = page
    if filename:
        querystring['filename'] = filename
    if source:
        querystring['source'] = source
    if name:
        querystring['name'] = name
    if mannequin_piece:
        querystring['mannequin_piece'] = mannequin_piece
    if variation:
        querystring['variation'] = variation
    if style:
        querystring['style'] = style
    if miles_price:
        querystring['miles_price'] = miles_price
    if color_1:
        querystring['color_1'] = color_1
    if internal_id:
        querystring['internal_id'] = internal_id
    if name_like:
        querystring['name__like'] = name_like
    if source_notes:
        querystring['source_notes'] = source_notes
    if villager_equippable:
        querystring['villager_equippable'] = villager_equippable
    if version:
        querystring['version'] = version
    if seasonal_availability:
        querystring['seasonal_availability'] = seasonal_availability
    if sell:
        querystring['sell'] = sell
    if color_2:
        querystring['color_2'] = color_2
    if diy:
        querystring['diy'] = diy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_accessories_api_v1_accessories_get(page_size: int=50, version: str=None, villager_equippable: str=None, filename: str=None, page: int=1, unique_entry_id: str=None, type: str=None, seasonal_availability: str=None, source: str=None, variation: str=None, diy: str=None, style: str=None, label_themes: str=None, internal_id: str=None, buy: str=None, size: str=None, mannequin_piece: str=None, name: str=None, label_themes_in: str=None, source_notes: str=None, miles_price: str=None, color_2: str=None, name_ilike: str=None, name_neq: str=None, sell: str=None, name_like: str=None, color_1: str=None, catalog: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/accessories"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if version:
        querystring['version'] = version
    if villager_equippable:
        querystring['villager_equippable'] = villager_equippable
    if filename:
        querystring['filename'] = filename
    if page:
        querystring['page'] = page
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if type:
        querystring['type'] = type
    if seasonal_availability:
        querystring['seasonal_availability'] = seasonal_availability
    if source:
        querystring['source'] = source
    if variation:
        querystring['variation'] = variation
    if diy:
        querystring['diy'] = diy
    if style:
        querystring['style'] = style
    if label_themes:
        querystring['label_themes'] = label_themes
    if internal_id:
        querystring['internal_id'] = internal_id
    if buy:
        querystring['buy'] = buy
    if size:
        querystring['size'] = size
    if mannequin_piece:
        querystring['mannequin_piece'] = mannequin_piece
    if name:
        querystring['name'] = name
    if label_themes_in:
        querystring['label_themes__in'] = label_themes_in
    if source_notes:
        querystring['source_notes'] = source_notes
    if miles_price:
        querystring['miles_price'] = miles_price
    if color_2:
        querystring['color_2'] = color_2
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if name_neq:
        querystring['name__neq'] = name_neq
    if sell:
        querystring['sell'] = sell
    if name_like:
        querystring['name__like'] = name_like
    if color_1:
        querystring['color_1'] = color_1
    if catalog:
        querystring['catalog'] = catalog
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_photos_api_v1_photos_get(variant_id: str=None, body_title: str=None, name_like: str=None, variation: str=None, unique_entry_id: str=None, filename: str=None, kit_cost: str=None, diy: str=None, source: str=None, catalog: str=None, name: str=None, sell: str=None, customize: str=None, size: str=None, pattern_title: str=None, page: int=1, color_1: str=None, color_2: str=None, page_size: int=50, name_ilike: str=None, internal_id: str=None, name_neq: str=None, pattern: str=None, buy: str=None, version: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/photos"
    querystring = {}
    if variant_id:
        querystring['variant_id'] = variant_id
    if body_title:
        querystring['body_title'] = body_title
    if name_like:
        querystring['name__like'] = name_like
    if variation:
        querystring['variation'] = variation
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if filename:
        querystring['filename'] = filename
    if kit_cost:
        querystring['kit_cost'] = kit_cost
    if diy:
        querystring['diy'] = diy
    if source:
        querystring['source'] = source
    if catalog:
        querystring['catalog'] = catalog
    if name:
        querystring['name'] = name
    if sell:
        querystring['sell'] = sell
    if customize:
        querystring['customize'] = customize
    if size:
        querystring['size'] = size
    if pattern_title:
        querystring['pattern_title'] = pattern_title
    if page:
        querystring['page'] = page
    if color_1:
        querystring['color_1'] = color_1
    if color_2:
        querystring['color_2'] = color_2
    if page_size:
        querystring['page_size'] = page_size
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if internal_id:
        querystring['internal_id'] = internal_id
    if name_neq:
        querystring['name__neq'] = name_neq
    if pattern:
        querystring['pattern'] = pattern
    if buy:
        querystring['buy'] = buy
    if version:
        querystring['version'] = version
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_art_api_v1_art_get(name_neq: str=None, speaker_type: str=None, real_artwork_title: str=None, hha_series: str=None, hha_concept_1: str=None, hha_set: str=None, source: str=None, interact: str=None, name: str=None, lighting_type: str=None, color_1: str=None, filename: str=None, source_notes: str=None, color_2: str=None, sell: str=None, tag: str=None, genuine: str=None, catalog: str=None, hha_concept_2: str=None, size: str=None, buy: str=None, internal_id: str=None, version: str=None, artist: str=None, name_like: str=None, category: str=None, page_size: int=50, unique_entry_id: str=None, museum_description: str=None, name_ilike: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/art"
    querystring = {}
    if name_neq:
        querystring['name__neq'] = name_neq
    if speaker_type:
        querystring['speaker_type'] = speaker_type
    if real_artwork_title:
        querystring['real_artwork_title'] = real_artwork_title
    if hha_series:
        querystring['hha_series'] = hha_series
    if hha_concept_1:
        querystring['hha_concept_1'] = hha_concept_1
    if hha_set:
        querystring['hha_set'] = hha_set
    if source:
        querystring['source'] = source
    if interact:
        querystring['interact'] = interact
    if name:
        querystring['name'] = name
    if lighting_type:
        querystring['lighting_type'] = lighting_type
    if color_1:
        querystring['color_1'] = color_1
    if filename:
        querystring['filename'] = filename
    if source_notes:
        querystring['source_notes'] = source_notes
    if color_2:
        querystring['color_2'] = color_2
    if sell:
        querystring['sell'] = sell
    if tag:
        querystring['tag'] = tag
    if genuine:
        querystring['genuine'] = genuine
    if catalog:
        querystring['catalog'] = catalog
    if hha_concept_2:
        querystring['hha_concept_2'] = hha_concept_2
    if size:
        querystring['size'] = size
    if buy:
        querystring['buy'] = buy
    if internal_id:
        querystring['internal_id'] = internal_id
    if version:
        querystring['version'] = version
    if artist:
        querystring['artist'] = artist
    if name_like:
        querystring['name__like'] = name_like
    if category:
        querystring['category'] = category
    if page_size:
        querystring['page_size'] = page_size
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if museum_description:
        querystring['museum_description'] = museum_description
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_posters_api_v1_posters_get(version: str=None, internal_id: str=None, sell: str=None, filename: str=None, catalog: str=None, name_like: str=None, name: str=None, color_2: str=None, unique_entry_id: str=None, page_size: int=50, name_neq: str=None, color_1: str=None, name_ilike: str=None, buy: str=None, source: str=None, page: int=1, source_notes: str=None, size: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/posters"
    querystring = {}
    if version:
        querystring['version'] = version
    if internal_id:
        querystring['internal_id'] = internal_id
    if sell:
        querystring['sell'] = sell
    if filename:
        querystring['filename'] = filename
    if catalog:
        querystring['catalog'] = catalog
    if name_like:
        querystring['name__like'] = name_like
    if name:
        querystring['name'] = name
    if color_2:
        querystring['color_2'] = color_2
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if page_size:
        querystring['page_size'] = page_size
    if name_neq:
        querystring['name__neq'] = name_neq
    if color_1:
        querystring['color_1'] = color_1
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if buy:
        querystring['buy'] = buy
    if source:
        querystring['source'] = source
    if page:
        querystring['page'] = page
    if source_notes:
        querystring['source_notes'] = source_notes
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_bottoms_api_v1_bottoms_get(version: str=None, name_ilike: str=None, name: str=None, size: str=None, label_themes: str=None, page: int=1, color_1: str=None, variation: str=None, source_notes: str=None, seasonal_availability: str=None, name_neq: str=None, color_2: str=None, unique_entry_id: str=None, internal_id: str=None, buy: str=None, source: str=None, filename: str=None, style: str=None, villager_equippable: str=None, name_like: str=None, diy: str=None, catalog: str=None, page_size: int=50, mannequin_piece: str=None, sell: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/bottoms"
    querystring = {}
    if version:
        querystring['version'] = version
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if name:
        querystring['name'] = name
    if size:
        querystring['size'] = size
    if label_themes:
        querystring['label_themes'] = label_themes
    if page:
        querystring['page'] = page
    if color_1:
        querystring['color_1'] = color_1
    if variation:
        querystring['variation'] = variation
    if source_notes:
        querystring['source_notes'] = source_notes
    if seasonal_availability:
        querystring['seasonal_availability'] = seasonal_availability
    if name_neq:
        querystring['name__neq'] = name_neq
    if color_2:
        querystring['color_2'] = color_2
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if internal_id:
        querystring['internal_id'] = internal_id
    if buy:
        querystring['buy'] = buy
    if source:
        querystring['source'] = source
    if filename:
        querystring['filename'] = filename
    if style:
        querystring['style'] = style
    if villager_equippable:
        querystring['villager_equippable'] = villager_equippable
    if name_like:
        querystring['name__like'] = name_like
    if diy:
        querystring['diy'] = diy
    if catalog:
        querystring['catalog'] = catalog
    if page_size:
        querystring['page_size'] = page_size
    if mannequin_piece:
        querystring['mannequin_piece'] = mannequin_piece
    if sell:
        querystring['sell'] = sell
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rugs_api_v1_rugs_get(source_notes: str=None, source: str=None, size: str=None, hha_concept_2: str=None, tag: str=None, unique_entry_id: str=None, sell: str=None, name_like: str=None, name_ilike: str=None, filename: str=None, internal_id: str=None, miles_price: str=None, hha_series: str=None, page: int=1, page_size: int=50, hha_concept_1: str=None, buy: str=None, version: str=None, diy: str=None, name_neq: str=None, catalog: str=None, name: str=None, color_1: str=None, color_2: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/rugs"
    querystring = {}
    if source_notes:
        querystring['source_notes'] = source_notes
    if source:
        querystring['source'] = source
    if size:
        querystring['size'] = size
    if hha_concept_2:
        querystring['hha_concept_2'] = hha_concept_2
    if tag:
        querystring['tag'] = tag
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if sell:
        querystring['sell'] = sell
    if name_like:
        querystring['name__like'] = name_like
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if filename:
        querystring['filename'] = filename
    if internal_id:
        querystring['internal_id'] = internal_id
    if miles_price:
        querystring['miles_price'] = miles_price
    if hha_series:
        querystring['hha_series'] = hha_series
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    if hha_concept_1:
        querystring['hha_concept_1'] = hha_concept_1
    if buy:
        querystring['buy'] = buy
    if version:
        querystring['version'] = version
    if diy:
        querystring['diy'] = diy
    if name_neq:
        querystring['name__neq'] = name_neq
    if catalog:
        querystring['catalog'] = catalog
    if name:
        querystring['name'] = name
    if color_1:
        querystring['color_1'] = color_1
    if color_2:
        querystring['color_2'] = color_2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_housewares_api_v1_housewares_get(page_size: int=50, color_1: str=None, miles_price: str=None, color_2: str=None, body_customize: str=None, source_notes: str=None, hha_set: str=None, outdoor: str=None, hha_series: str=None, lighting_type: str=None, buy: str=None, pattern: str=None, internal_id: str=None, hha_concept_2: str=None, name: str=None, unique_entry_id: str=None, hha_concept_1: str=None, name_neq: str=None, kit_cost: str=None, version: str=None, size: str=None, source: str=None, diy: str=None, name_ilike: str=None, filename: str=None, sell: str=None, tag: str=None, variation: str=None, pattern_title: str=None, speaker_type: str=None, variant_id: str=None, interact: str=None, pattern_customize: str=None, page: int=1, name_like: str=None, catalog: str=None, body_title: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/housewares"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if color_1:
        querystring['color_1'] = color_1
    if miles_price:
        querystring['miles_price'] = miles_price
    if color_2:
        querystring['color_2'] = color_2
    if body_customize:
        querystring['body_customize'] = body_customize
    if source_notes:
        querystring['source_notes'] = source_notes
    if hha_set:
        querystring['hha_set'] = hha_set
    if outdoor:
        querystring['outdoor'] = outdoor
    if hha_series:
        querystring['hha_series'] = hha_series
    if lighting_type:
        querystring['lighting_type'] = lighting_type
    if buy:
        querystring['buy'] = buy
    if pattern:
        querystring['pattern'] = pattern
    if internal_id:
        querystring['internal_id'] = internal_id
    if hha_concept_2:
        querystring['hha_concept_2'] = hha_concept_2
    if name:
        querystring['name'] = name
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if hha_concept_1:
        querystring['hha_concept_1'] = hha_concept_1
    if name_neq:
        querystring['name__neq'] = name_neq
    if kit_cost:
        querystring['kit_cost'] = kit_cost
    if version:
        querystring['version'] = version
    if size:
        querystring['size'] = size
    if source:
        querystring['source'] = source
    if diy:
        querystring['diy'] = diy
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if filename:
        querystring['filename'] = filename
    if sell:
        querystring['sell'] = sell
    if tag:
        querystring['tag'] = tag
    if variation:
        querystring['variation'] = variation
    if pattern_title:
        querystring['pattern_title'] = pattern_title
    if speaker_type:
        querystring['speaker_type'] = speaker_type
    if variant_id:
        querystring['variant_id'] = variant_id
    if interact:
        querystring['interact'] = interact
    if pattern_customize:
        querystring['pattern_customize'] = pattern_customize
    if page:
        querystring['page'] = page
    if name_like:
        querystring['name__like'] = name_like
    if catalog:
        querystring['catalog'] = catalog
    if body_title:
        querystring['body_title'] = body_title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_miscellaneous_api_v1_miscellaneous_get(variation: str=None, pattern: str=None, color_1: str=None, page: int=1, source_notes: str=None, hha_set: str=None, hha_concept_2: str=None, name_ilike: str=None, name_neq: str=None, buy: str=None, body_title: str=None, size: str=None, speaker_type: str=None, diy: str=None, color_2: str=None, unique_entry_id: str=None, lighting_type: str=None, source: str=None, body_customize: str=None, outdoor: str=None, name: str=None, sell: str=None, kit_cost: str=None, version: str=None, name_like: str=None, interact: str=None, pattern_title: str=None, internal_id: str=None, filename: str=None, page_size: int=50, variant_id: str=None, hha_series: str=None, catalog: str=None, tag: str=None, hha_concept_1: str=None, pattern_customize: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/miscellaneous"
    querystring = {}
    if variation:
        querystring['variation'] = variation
    if pattern:
        querystring['pattern'] = pattern
    if color_1:
        querystring['color_1'] = color_1
    if page:
        querystring['page'] = page
    if source_notes:
        querystring['source_notes'] = source_notes
    if hha_set:
        querystring['hha_set'] = hha_set
    if hha_concept_2:
        querystring['hha_concept_2'] = hha_concept_2
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if name_neq:
        querystring['name__neq'] = name_neq
    if buy:
        querystring['buy'] = buy
    if body_title:
        querystring['body_title'] = body_title
    if size:
        querystring['size'] = size
    if speaker_type:
        querystring['speaker_type'] = speaker_type
    if diy:
        querystring['diy'] = diy
    if color_2:
        querystring['color_2'] = color_2
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if lighting_type:
        querystring['lighting_type'] = lighting_type
    if source:
        querystring['source'] = source
    if body_customize:
        querystring['body_customize'] = body_customize
    if outdoor:
        querystring['outdoor'] = outdoor
    if name:
        querystring['name'] = name
    if sell:
        querystring['sell'] = sell
    if kit_cost:
        querystring['kit_cost'] = kit_cost
    if version:
        querystring['version'] = version
    if name_like:
        querystring['name__like'] = name_like
    if interact:
        querystring['interact'] = interact
    if pattern_title:
        querystring['pattern_title'] = pattern_title
    if internal_id:
        querystring['internal_id'] = internal_id
    if filename:
        querystring['filename'] = filename
    if page_size:
        querystring['page_size'] = page_size
    if variant_id:
        querystring['variant_id'] = variant_id
    if hha_series:
        querystring['hha_series'] = hha_series
    if catalog:
        querystring['catalog'] = catalog
    if tag:
        querystring['tag'] = tag
    if hha_concept_1:
        querystring['hha_concept_1'] = hha_concept_1
    if pattern_customize:
        querystring['pattern_customize'] = pattern_customize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_shoes_api_v1_shoes_get(label_themes: str=None, name_ilike: str=None, style: str=None, villager_equippable: str=None, source: str=None, color_2: str=None, diy: str=None, filename: str=None, color_1: str=None, page_size: int=50, unique_entry_id: str=None, mannequin_piece: str=None, name_neq: str=None, name: str=None, catalog: str=None, internal_id: str=None, version: str=None, buy: str=None, source_notes: str=None, miles_price: str=None, page: int=1, variation: str=None, size: str=None, seasonal_availability: str=None, name_like: str=None, sell: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/shoes"
    querystring = {}
    if label_themes:
        querystring['label_themes'] = label_themes
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if style:
        querystring['style'] = style
    if villager_equippable:
        querystring['villager_equippable'] = villager_equippable
    if source:
        querystring['source'] = source
    if color_2:
        querystring['color_2'] = color_2
    if diy:
        querystring['diy'] = diy
    if filename:
        querystring['filename'] = filename
    if color_1:
        querystring['color_1'] = color_1
    if page_size:
        querystring['page_size'] = page_size
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if mannequin_piece:
        querystring['mannequin_piece'] = mannequin_piece
    if name_neq:
        querystring['name__neq'] = name_neq
    if name:
        querystring['name'] = name
    if catalog:
        querystring['catalog'] = catalog
    if internal_id:
        querystring['internal_id'] = internal_id
    if version:
        querystring['version'] = version
    if buy:
        querystring['buy'] = buy
    if source_notes:
        querystring['source_notes'] = source_notes
    if miles_price:
        querystring['miles_price'] = miles_price
    if page:
        querystring['page'] = page
    if variation:
        querystring['variation'] = variation
    if size:
        querystring['size'] = size
    if seasonal_availability:
        querystring['seasonal_availability'] = seasonal_availability
    if name_like:
        querystring['name__like'] = name_like
    if sell:
        querystring['sell'] = sell
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tops_api_v1_tops_get(name_neq: str=None, name_ilike: str=None, catalog: str=None, size: str=None, buy: str=None, variation: str=None, page: int=1, source: str=None, style: str=None, filename: str=None, label_themes: str=None, name_like: str=None, color_1: str=None, sell: str=None, villager_equippable: str=None, unique_entry_id: str=None, diy: str=None, source_notes: str=None, internal_id: str=None, miles_price: str=None, version: str=None, mannequin_piece: str=None, seasonal_availability: str=None, color_2: str=None, page_size: int=50, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/tops"
    querystring = {}
    if name_neq:
        querystring['name__neq'] = name_neq
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if catalog:
        querystring['catalog'] = catalog
    if size:
        querystring['size'] = size
    if buy:
        querystring['buy'] = buy
    if variation:
        querystring['variation'] = variation
    if page:
        querystring['page'] = page
    if source:
        querystring['source'] = source
    if style:
        querystring['style'] = style
    if filename:
        querystring['filename'] = filename
    if label_themes:
        querystring['label_themes'] = label_themes
    if name_like:
        querystring['name__like'] = name_like
    if color_1:
        querystring['color_1'] = color_1
    if sell:
        querystring['sell'] = sell
    if villager_equippable:
        querystring['villager_equippable'] = villager_equippable
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if diy:
        querystring['diy'] = diy
    if source_notes:
        querystring['source_notes'] = source_notes
    if internal_id:
        querystring['internal_id'] = internal_id
    if miles_price:
        querystring['miles_price'] = miles_price
    if version:
        querystring['version'] = version
    if mannequin_piece:
        querystring['mannequin_piece'] = mannequin_piece
    if seasonal_availability:
        querystring['seasonal_availability'] = seasonal_availability
    if color_2:
        querystring['color_2'] = color_2
    if page_size:
        querystring['page_size'] = page_size
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_floors_api_v1_floors_get(hha_concept_1: str=None, vfx: str=None, catalog: str=None, sell: str=None, source: str=None, version: str=None, unique_entry_id: str=None, page: int=1, name_like: str=None, hha_concept_2: str=None, miles_price: str=None, page_size: int=50, hha_series: str=None, tag: str=None, name: str=None, internal_id: str=None, name_neq: str=None, diy: str=None, filename: str=None, buy: str=None, source_notes: str=None, color_1: str=None, color_2: str=None, name_ilike: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/floors"
    querystring = {}
    if hha_concept_1:
        querystring['hha_concept_1'] = hha_concept_1
    if vfx:
        querystring['vfx'] = vfx
    if catalog:
        querystring['catalog'] = catalog
    if sell:
        querystring['sell'] = sell
    if source:
        querystring['source'] = source
    if version:
        querystring['version'] = version
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if page:
        querystring['page'] = page
    if name_like:
        querystring['name__like'] = name_like
    if hha_concept_2:
        querystring['hha_concept_2'] = hha_concept_2
    if miles_price:
        querystring['miles_price'] = miles_price
    if page_size:
        querystring['page_size'] = page_size
    if hha_series:
        querystring['hha_series'] = hha_series
    if tag:
        querystring['tag'] = tag
    if name:
        querystring['name'] = name
    if internal_id:
        querystring['internal_id'] = internal_id
    if name_neq:
        querystring['name__neq'] = name_neq
    if diy:
        querystring['diy'] = diy
    if filename:
        querystring['filename'] = filename
    if buy:
        querystring['buy'] = buy
    if source_notes:
        querystring['source_notes'] = source_notes
    if color_1:
        querystring['color_1'] = color_1
    if color_2:
        querystring['color_2'] = color_2
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fish_api_v1_fish_get(sh_dec: str=None, sh_jan: str=None, nh_dec: str=None, sh_oct: str=None, name_neq: str=None, name_ilike: str=None, sh_jun: str=None, size: str=None, page_size: int=50, sh_aug: str=None, nh_jul: str=None, nh_jan: str=None, nh_apr: str=None, color_2: str=None, name: str=None, nh_oct: str=None, rain_snow_catch_up: str=None, name_like: str=None, nh_jun: str=None, unique_entry_id: str=None, sh_jul: str=None, total_catches_to_unlock: str=None, sh_may: str=None, lighting_type: str=None, nh_aug: str=None, shadow: str=None, nh_sep: str=None, furniture_filename: str=None, sh_apr: str=None, sh_mar: str=None, sell: str=None, spawn_rates: str=None, sh_sep: str=None, page: int=1, color_1: str=None, sh_nov: str=None, where_how: str=None, icon_filename: str=None, nh_may: str=None, nh_feb: str=None, nh_mar: str=None, nh_nov: str=None, critterpedia_filename: str=None, sh_feb: str=None, internal_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/fish"
    querystring = {}
    if sh_dec:
        querystring['sh_dec'] = sh_dec
    if sh_jan:
        querystring['sh_jan'] = sh_jan
    if nh_dec:
        querystring['nh_dec'] = nh_dec
    if sh_oct:
        querystring['sh_oct'] = sh_oct
    if name_neq:
        querystring['name__neq'] = name_neq
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if sh_jun:
        querystring['sh_jun'] = sh_jun
    if size:
        querystring['size'] = size
    if page_size:
        querystring['page_size'] = page_size
    if sh_aug:
        querystring['sh_aug'] = sh_aug
    if nh_jul:
        querystring['nh_jul'] = nh_jul
    if nh_jan:
        querystring['nh_jan'] = nh_jan
    if nh_apr:
        querystring['nh_apr'] = nh_apr
    if color_2:
        querystring['color_2'] = color_2
    if name:
        querystring['name'] = name
    if nh_oct:
        querystring['nh_oct'] = nh_oct
    if rain_snow_catch_up:
        querystring['rain_snow_catch_up'] = rain_snow_catch_up
    if name_like:
        querystring['name__like'] = name_like
    if nh_jun:
        querystring['nh_jun'] = nh_jun
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if sh_jul:
        querystring['sh_jul'] = sh_jul
    if total_catches_to_unlock:
        querystring['total_catches_to_unlock'] = total_catches_to_unlock
    if sh_may:
        querystring['sh_may'] = sh_may
    if lighting_type:
        querystring['lighting_type'] = lighting_type
    if nh_aug:
        querystring['nh_aug'] = nh_aug
    if shadow:
        querystring['shadow'] = shadow
    if nh_sep:
        querystring['nh_sep'] = nh_sep
    if furniture_filename:
        querystring['furniture_filename'] = furniture_filename
    if sh_apr:
        querystring['sh_apr'] = sh_apr
    if sh_mar:
        querystring['sh_mar'] = sh_mar
    if sell:
        querystring['sell'] = sell
    if spawn_rates:
        querystring['spawn_rates'] = spawn_rates
    if sh_sep:
        querystring['sh_sep'] = sh_sep
    if page:
        querystring['page'] = page
    if color_1:
        querystring['color_1'] = color_1
    if sh_nov:
        querystring['sh_nov'] = sh_nov
    if where_how:
        querystring['where_how'] = where_how
    if icon_filename:
        querystring['icon_filename'] = icon_filename
    if nh_may:
        querystring['nh_may'] = nh_may
    if nh_feb:
        querystring['nh_feb'] = nh_feb
    if nh_mar:
        querystring['nh_mar'] = nh_mar
    if nh_nov:
        querystring['nh_nov'] = nh_nov
    if critterpedia_filename:
        querystring['critterpedia_filename'] = critterpedia_filename
    if sh_feb:
        querystring['sh_feb'] = sh_feb
    if internal_id:
        querystring['internal_id'] = internal_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fencing_api_v1_fencing_get(buy: str=None, name: str=None, unique_entry_id: str=None, sell: str=None, page_size: int=50, source_notes: str=None, name_like: str=None, diy: str=None, stack_size: str=None, source: str=None, name_ilike: str=None, page: int=1, version: str=None, name_neq: str=None, internal_id: str=None, filename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/fencing"
    querystring = {}
    if buy:
        querystring['buy'] = buy
    if name:
        querystring['name'] = name
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if sell:
        querystring['sell'] = sell
    if page_size:
        querystring['page_size'] = page_size
    if source_notes:
        querystring['source_notes'] = source_notes
    if name_like:
        querystring['name__like'] = name_like
    if diy:
        querystring['diy'] = diy
    if stack_size:
        querystring['stack_size'] = stack_size
    if source:
        querystring['source'] = source
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if page:
        querystring['page'] = page
    if version:
        querystring['version'] = version
    if name_neq:
        querystring['name__neq'] = name_neq
    if internal_id:
        querystring['internal_id'] = internal_id
    if filename:
        querystring['filename'] = filename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_construction_api_v1_construction_get(name_neq: str=None, name_ilike: str=None, name_like: str=None, source: str=None, buy: str=None, page: int=1, category: str=None, page_size: int=50, version: str=None, filename: str=None, unique_entry_id: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/construction"
    querystring = {}
    if name_neq:
        querystring['name__neq'] = name_neq
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if name_like:
        querystring['name__like'] = name_like
    if source:
        querystring['source'] = source
    if buy:
        querystring['buy'] = buy
    if page:
        querystring['page'] = page
    if category:
        querystring['category'] = category
    if page_size:
        querystring['page_size'] = page_size
    if version:
        querystring['version'] = version
    if filename:
        querystring['filename'] = filename
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_other_api_v1_other_get(buy: str=None, name_like: str=None, color_2: str=None, unique_entry_id: str=None, miles_price: str=None, tag: str=None, page_size: int=50, page: int=1, version: str=None, color_1: str=None, source: str=None, stack_size: str=None, diy: str=None, name_neq: str=None, source_notes: str=None, internal_id: str=None, name: str=None, filename: str=None, sell: str=None, name_ilike: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/other"
    querystring = {}
    if buy:
        querystring['buy'] = buy
    if name_like:
        querystring['name__like'] = name_like
    if color_2:
        querystring['color_2'] = color_2
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if miles_price:
        querystring['miles_price'] = miles_price
    if tag:
        querystring['tag'] = tag
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    if version:
        querystring['version'] = version
    if color_1:
        querystring['color_1'] = color_1
    if source:
        querystring['source'] = source
    if stack_size:
        querystring['stack_size'] = stack_size
    if diy:
        querystring['diy'] = diy
    if name_neq:
        querystring['name__neq'] = name_neq
    if source_notes:
        querystring['source_notes'] = source_notes
    if internal_id:
        querystring['internal_id'] = internal_id
    if name:
        querystring['name'] = name
    if filename:
        querystring['filename'] = filename
    if sell:
        querystring['sell'] = sell
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_bags_api_v1_bags_get(internal_id: str=None, unique_entry_id: str=None, name_ilike: str=None, variation: str=None, size: str=None, name_like: str=None, name: str=None, villager_equippable: str=None, source: str=None, color_1: str=None, filename: str=None, seasonal_availability: str=None, buy: str=None, sell: str=None, page_size: int=50, style: str=None, page: int=1, source_notes: str=None, color_2: str=None, version: str=None, catalog: str=None, label_themes: str=None, miles_price: str=None, diy: str=None, name_neq: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/bags"
    querystring = {}
    if internal_id:
        querystring['internal_id'] = internal_id
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if variation:
        querystring['variation'] = variation
    if size:
        querystring['size'] = size
    if name_like:
        querystring['name__like'] = name_like
    if name:
        querystring['name'] = name
    if villager_equippable:
        querystring['villager_equippable'] = villager_equippable
    if source:
        querystring['source'] = source
    if color_1:
        querystring['color_1'] = color_1
    if filename:
        querystring['filename'] = filename
    if seasonal_availability:
        querystring['seasonal_availability'] = seasonal_availability
    if buy:
        querystring['buy'] = buy
    if sell:
        querystring['sell'] = sell
    if page_size:
        querystring['page_size'] = page_size
    if style:
        querystring['style'] = style
    if page:
        querystring['page'] = page
    if source_notes:
        querystring['source_notes'] = source_notes
    if color_2:
        querystring['color_2'] = color_2
    if version:
        querystring['version'] = version
    if catalog:
        querystring['catalog'] = catalog
    if label_themes:
        querystring['label_themes'] = label_themes
    if miles_price:
        querystring['miles_price'] = miles_price
    if diy:
        querystring['diy'] = diy
    if name_neq:
        querystring['name__neq'] = name_neq
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music_api_v1_music_get(buy: str=None, name_like: str=None, source_notes: str=None, color_2: str=None, color_1: str=None, name: str=None, unique_entry_id: str=None, name_ilike: str=None, internal_id: str=None, filename: str=None, sell: str=None, page_size: int=50, size: str=None, page: int=1, source: str=None, catalog: str=None, version: str=None, name_neq: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/music"
    querystring = {}
    if buy:
        querystring['buy'] = buy
    if name_like:
        querystring['name__like'] = name_like
    if source_notes:
        querystring['source_notes'] = source_notes
    if color_2:
        querystring['color_2'] = color_2
    if color_1:
        querystring['color_1'] = color_1
    if name:
        querystring['name'] = name
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if internal_id:
        querystring['internal_id'] = internal_id
    if filename:
        querystring['filename'] = filename
    if sell:
        querystring['sell'] = sell
    if page_size:
        querystring['page_size'] = page_size
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if source:
        querystring['source'] = source
    if catalog:
        querystring['catalog'] = catalog
    if version:
        querystring['version'] = version
    if name_neq:
        querystring['name__neq'] = name_neq
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recipes_api_v1_recipes_get(num_mat_6: str=None, category: str=None, material_3: str=None, internal_id: str=None, material_6: str=None, name: str=None, name_neq: str=None, recipes_to_unlock: str=None, unique_entry_id: str=None, name_like: str=None, miles_price: str=None, source: str=None, num_mat_4: str=None, num_mat_3: str=None, page_size: int=50, version: str=None, sell: str=None, num_mat_5: str=None, serial_id: str=None, material_2: str=None, num_mat_1: str=None, material_1: str=None, name_ilike: str=None, material_4: str=None, material_5: str=None, source_notes: str=None, page: int=1, num_mat_2: str=None, buy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/recipes"
    querystring = {}
    if num_mat_6:
        querystring['num_mat_6'] = num_mat_6
    if category:
        querystring['category'] = category
    if material_3:
        querystring['material_3'] = material_3
    if internal_id:
        querystring['internal_id'] = internal_id
    if material_6:
        querystring['material_6'] = material_6
    if name:
        querystring['name'] = name
    if name_neq:
        querystring['name__neq'] = name_neq
    if recipes_to_unlock:
        querystring['recipes_to_unlock'] = recipes_to_unlock
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if name_like:
        querystring['name__like'] = name_like
    if miles_price:
        querystring['miles_price'] = miles_price
    if source:
        querystring['source'] = source
    if num_mat_4:
        querystring['num_mat_4'] = num_mat_4
    if num_mat_3:
        querystring['num_mat_3'] = num_mat_3
    if page_size:
        querystring['page_size'] = page_size
    if version:
        querystring['version'] = version
    if sell:
        querystring['sell'] = sell
    if num_mat_5:
        querystring['num_mat_5'] = num_mat_5
    if serial_id:
        querystring['serial_id'] = serial_id
    if material_2:
        querystring['material_2'] = material_2
    if num_mat_1:
        querystring['num_mat_1'] = num_mat_1
    if material_1:
        querystring['material_1'] = material_1
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if material_4:
        querystring['material_4'] = material_4
    if material_5:
        querystring['material_5'] = material_5
    if source_notes:
        querystring['source_notes'] = source_notes
    if page:
        querystring['page'] = page
    if num_mat_2:
        querystring['num_mat_2'] = num_mat_2
    if buy:
        querystring['buy'] = buy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_wallpaper_api_v1_wallpaper_get(hha_series: str=None, buy: str=None, tag: str=None, source: str=None, hha_concept_2: str=None, unique_entry_id: str=None, name_ilike: str=None, sell: str=None, curtain_type: str=None, color_2: str=None, name_neq: str=None, pane_type: str=None, diy: str=None, source_notes: str=None, curtain_color: str=None, miles_price: str=None, window_color: str=None, catalog: str=None, color_1: str=None, name_like: str=None, page: int=1, vfx_type: str=None, hha_concept_1: str=None, internal_id: str=None, ceiling_type: str=None, vfx: str=None, window_type: str=None, page_size: int=50, filename: str=None, version: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/wallpaper"
    querystring = {}
    if hha_series:
        querystring['hha_series'] = hha_series
    if buy:
        querystring['buy'] = buy
    if tag:
        querystring['tag'] = tag
    if source:
        querystring['source'] = source
    if hha_concept_2:
        querystring['hha_concept_2'] = hha_concept_2
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if sell:
        querystring['sell'] = sell
    if curtain_type:
        querystring['curtain_type'] = curtain_type
    if color_2:
        querystring['color_2'] = color_2
    if name_neq:
        querystring['name__neq'] = name_neq
    if pane_type:
        querystring['pane_type'] = pane_type
    if diy:
        querystring['diy'] = diy
    if source_notes:
        querystring['source_notes'] = source_notes
    if curtain_color:
        querystring['curtain_color'] = curtain_color
    if miles_price:
        querystring['miles_price'] = miles_price
    if window_color:
        querystring['window_color'] = window_color
    if catalog:
        querystring['catalog'] = catalog
    if color_1:
        querystring['color_1'] = color_1
    if name_like:
        querystring['name__like'] = name_like
    if page:
        querystring['page'] = page
    if vfx_type:
        querystring['vfx_type'] = vfx_type
    if hha_concept_1:
        querystring['hha_concept_1'] = hha_concept_1
    if internal_id:
        querystring['internal_id'] = internal_id
    if ceiling_type:
        querystring['ceiling_type'] = ceiling_type
    if vfx:
        querystring['vfx'] = vfx
    if window_type:
        querystring['window_type'] = window_type
    if page_size:
        querystring['page_size'] = page_size
    if filename:
        querystring['filename'] = filename
    if version:
        querystring['version'] = version
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_umbrellas_api_v1_umbrellas_get(filename: str=None, name_neq: str=None, name: str=None, name_ilike: str=None, page_size: int=50, source_notes: str=None, buy: str=None, color_1: str=None, diy: str=None, internal_id: str=None, unique_entry_id: str=None, source: str=None, name_like: str=None, villager_equippable: str=None, page: int=1, size: str=None, catalog: str=None, sell: str=None, miles_price: str=None, color_2: str=None, version: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/umbrellas"
    querystring = {}
    if filename:
        querystring['filename'] = filename
    if name_neq:
        querystring['name__neq'] = name_neq
    if name:
        querystring['name'] = name
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if page_size:
        querystring['page_size'] = page_size
    if source_notes:
        querystring['source_notes'] = source_notes
    if buy:
        querystring['buy'] = buy
    if color_1:
        querystring['color_1'] = color_1
    if diy:
        querystring['diy'] = diy
    if internal_id:
        querystring['internal_id'] = internal_id
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if source:
        querystring['source'] = source
    if name_like:
        querystring['name__like'] = name_like
    if villager_equippable:
        querystring['villager_equippable'] = villager_equippable
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if catalog:
        querystring['catalog'] = catalog
    if sell:
        querystring['sell'] = sell
    if miles_price:
        querystring['miles_price'] = miles_price
    if color_2:
        querystring['color_2'] = color_2
    if version:
        querystring['version'] = version
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_villagers_api_v1_villagers_get(wallpaper: str=None, color_1: str=None, birthday: str=None, species: str=None, name_ilike: str=None, name: str=None, catchphrase: str=None, favorite_song: str=None, unique_entry_id: str=None, style_1: str=None, page: int=1, flooring: str=None, filename: str=None, personality: str=None, style_2: str=None, name_neq: str=None, page_size: int=50, color_2: str=None, name_like: str=None, furniture_list: str=None, hobby: str=None, gender: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/villagers"
    querystring = {}
    if wallpaper:
        querystring['wallpaper'] = wallpaper
    if color_1:
        querystring['color_1'] = color_1
    if birthday:
        querystring['birthday'] = birthday
    if species:
        querystring['species'] = species
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if name:
        querystring['name'] = name
    if catchphrase:
        querystring['catchphrase'] = catchphrase
    if favorite_song:
        querystring['favorite_song'] = favorite_song
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if style_1:
        querystring['style_1'] = style_1
    if page:
        querystring['page'] = page
    if flooring:
        querystring['flooring'] = flooring
    if filename:
        querystring['filename'] = filename
    if personality:
        querystring['personality'] = personality
    if style_2:
        querystring['style_2'] = style_2
    if name_neq:
        querystring['name__neq'] = name_neq
    if page_size:
        querystring['page_size'] = page_size
    if color_2:
        querystring['color_2'] = color_2
    if name_like:
        querystring['name__like'] = name_like
    if furniture_list:
        querystring['furniture_list'] = furniture_list
    if hobby:
        querystring['hobby'] = hobby
    if gender:
        querystring['gender'] = gender
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_insects_api_v1_insects_get(nh_may: str=None, sh_aug: str=None, sh_feb: str=None, furniture_filename: str=None, sh_nov: str=None, sh_dec: str=None, unique_entry_id: str=None, sh_sep: str=None, name_like: str=None, nh_nov: str=None, nh_jan: str=None, spawn_rates: str=None, nh_dec: str=None, nh_mar: str=None, nh_oct: str=None, nh_jul: str=None, internal_id: str=None, nh_apr: str=None, nh_jun: str=None, nh_aug: str=None, page_size: int=50, name: str=None, sh_may: str=None, weather: str=None, name_neq: str=None, where_how: str=None, nh_feb: str=None, sh_oct: str=None, nh_sep: str=None, sh_jul: str=None, icon_filename: str=None, sell: str=None, sh_apr: str=None, total_catches_to_unlock: str=None, sh_jan: str=None, page: int=1, color_2: str=None, sh_jun: str=None, critterpedia_filename: str=None, name_ilike: str=None, color_1: str=None, sh_mar: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://animal-crossing-new-horizons2.p.rapidapi.com/api/v1/insects"
    querystring = {}
    if nh_may:
        querystring['nh_may'] = nh_may
    if sh_aug:
        querystring['sh_aug'] = sh_aug
    if sh_feb:
        querystring['sh_feb'] = sh_feb
    if furniture_filename:
        querystring['furniture_filename'] = furniture_filename
    if sh_nov:
        querystring['sh_nov'] = sh_nov
    if sh_dec:
        querystring['sh_dec'] = sh_dec
    if unique_entry_id:
        querystring['unique_entry_id'] = unique_entry_id
    if sh_sep:
        querystring['sh_sep'] = sh_sep
    if name_like:
        querystring['name__like'] = name_like
    if nh_nov:
        querystring['nh_nov'] = nh_nov
    if nh_jan:
        querystring['nh_jan'] = nh_jan
    if spawn_rates:
        querystring['spawn_rates'] = spawn_rates
    if nh_dec:
        querystring['nh_dec'] = nh_dec
    if nh_mar:
        querystring['nh_mar'] = nh_mar
    if nh_oct:
        querystring['nh_oct'] = nh_oct
    if nh_jul:
        querystring['nh_jul'] = nh_jul
    if internal_id:
        querystring['internal_id'] = internal_id
    if nh_apr:
        querystring['nh_apr'] = nh_apr
    if nh_jun:
        querystring['nh_jun'] = nh_jun
    if nh_aug:
        querystring['nh_aug'] = nh_aug
    if page_size:
        querystring['page_size'] = page_size
    if name:
        querystring['name'] = name
    if sh_may:
        querystring['sh_may'] = sh_may
    if weather:
        querystring['weather'] = weather
    if name_neq:
        querystring['name__neq'] = name_neq
    if where_how:
        querystring['where_how'] = where_how
    if nh_feb:
        querystring['nh_feb'] = nh_feb
    if sh_oct:
        querystring['sh_oct'] = sh_oct
    if nh_sep:
        querystring['nh_sep'] = nh_sep
    if sh_jul:
        querystring['sh_jul'] = sh_jul
    if icon_filename:
        querystring['icon_filename'] = icon_filename
    if sell:
        querystring['sell'] = sell
    if sh_apr:
        querystring['sh_apr'] = sh_apr
    if total_catches_to_unlock:
        querystring['total_catches_to_unlock'] = total_catches_to_unlock
    if sh_jan:
        querystring['sh_jan'] = sh_jan
    if page:
        querystring['page'] = page
    if color_2:
        querystring['color_2'] = color_2
    if sh_jun:
        querystring['sh_jun'] = sh_jun
    if critterpedia_filename:
        querystring['critterpedia_filename'] = critterpedia_filename
    if name_ilike:
        querystring['name__ilike'] = name_ilike
    if color_1:
        querystring['color_1'] = color_1
    if sh_mar:
        querystring['sh_mar'] = sh_mar
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animal-crossing-new-horizons2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

