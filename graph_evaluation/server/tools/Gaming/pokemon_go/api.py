import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def type_effectiveness(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When a Pokemon attacks, a multiplier is applied based on the attacker and defender types. This type effectiveness is very important to help deal the most damage as possible in raids and pvp. This API details how each type multiplier is applied.
		
		Returns a JSON dict where each key is the attacking type and the value is a dict of defender types and the damage multiplier."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/type_effectiveness.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pokemon_rarity(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pokemon are split up into 3 rarity types. These are Standard, Legendary, and Mythic. Typically Legendary and Mythic Pokemon have different trading rules along with a much higher buddy walking distance. This API details how what rarity ranking Pokemon Go places each Pokemon in.
		
		Returns a JSON dict where each key is the rarity type and the value is a list of dicts containing the details of Pokemon in that rarity group. Each Pokemon dict has the pokemon_id, pokemon_name and rarity inside it for ease of processing."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/pokemon_rarity.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pokemon_power_up_requirements(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To power up a pokemon you need to pay some stardust and candy. Pokemon can be powered up to two levels higher than your level, to a max level of 40. This API details how much it will cost to power up a Pokemon each level.
		
		Returns a JSON dict where each key is the current level and the value is a dictionary containing the cost of powering up a Pokemon at that level. Each powerup dict contains the candy to upgrade, the stardust to upgrade, the current level and the next level."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/pokemon_powerup_requirements.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pokemon_genders(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each Pokemon either has a gender of male or female, or is classified as genderless. This is important for some Pokemon who have gender specific evolutions. For example only a female Combee will evolve.
		
		For each Pokemon there is percentage chance of any gender. For example some Pokemon are exclusively male or female and some are skewed towards one gender. This API details the gender breakdown for each Pokemon.
		
		Returns a JSON dict where each key is the name of the gender ratio and the value is a list of Pokemon and their gender percentages. This dictionary holds the Pokemon ID, Pokemon name, and their gender percentages under the gender key.
		
		The gender key is a dictionary which has three possible keys, female_percent, male_percent, genderless_percent. These values are a value from 0 to 1 and are not percentages but are kept named as such as this is what the game master calls it. 1 represents the Pokemon is entirely of one gender. The actual percentage may be obtained by multiplying the value by 100."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/pokemon_genders.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_pokemon_moves(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When a Pokemon is caught, evolved, or hatched the moves are randomly chosen from a pool of potential moves. These moves change from time to time as moves are added and removed from the pool of moves. Whenever a TM is used, a random move from the list of potential moves is chosen and the Pokemon will learn it in place of an old move.
		
		During community days and other events some Pokemon will be given special moves for short period of times. These moves will be added to the pool so the Pokemon can get them but are removed after the event finishes. These are called Legacy moves will typically be available by using an elite TM. These moves are noted in the elite_charged_moves and elite_fast_moves arrays. Currently an Elite TM can only be used to learn legacy charged moves.
		
		This API returns all moves that Pokemon can currently learn via catching, evolving, hatching, or using TM's.
		
		Returns a JSON array where each element is a dict containing Pokemon ID, Pokemon name, an array of charged moves, an array of fast moves, an array of charged moves learnable from an elite TM, an array of fast legacy fast moves, and optionally the form."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/current_pokemon_moves.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pokemon_evolutions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Some Pokemon can evolve into a stronger form by feeding them candy or meeting certain conditions. Once these conditions have been met you can press the "evolve" button to transform them to the next form.
		
		If multiple evolutions meet the requirements and they have the same priority (or it is not set) the evolution will be randomly chosen.
		
		This API lists all the Pokemon that can evolve, what they can evolve into, and the requirements for their evolution.
		
		Returns a JSON array of objects, the objects have the following keys.
		
		- pokemon_id - ID of the Pokemon that evolves
		- pokemon_name - Name of the Pokemon that evolves
		- form - Only present if the Pokemon has multiple forms
		- evolutions - An array of objects detailing what Pokemon this can evolve into. Each object has the following keys:
		
		- - candy_required - The amount of candy to evolve this Pokemon with standard means
		- - item_required - If the Pokemon requires an item to evolve this will be the name of the item
		- - lure_required - If the Pokemon requires a lure to evolve this will be the name of the lure
		- - no_candy_cost_if_traded - This will be set to true if the evolution will cost no candy after trading
		- - priority - If a priority is set then the evolution with the highest priority will be chosen above those with a lower priority when multiple evolution criteria are met
		- - only_evolves_in_daytime - Set to true if the Pokemon will only evolve in the daytime
		- - only_evolves_in_nighttime - Set to true if the Pokemon will only evolve at night time
		- - must_be_buddy_to_evolve - Set if this evolution can only occur if they are your buddy
		- - buddy_distance_required - Set if this Pokemon must have been walked a specific amount of distance to evolve
		- - gender_required - Set to Male/Female if a specific gender is required for the evolution"
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/pokemon_evolutions.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def raid_exclusive_pokemon(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a JSON dict with the keys being the Pokemon ID, the values are an array containing the Pokemon name, ID, and and level raid they can be found in.  Currently all raid exclusive pokemon are tied to a specific raid level. In the future the API might need to change if this changes."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/raid_exclusive_pokemon.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pokemon_buddy_distances(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When you make a Pokemon your buddy after a certain distance walked with them you will get 1 candy from them. This distance depends on the specific Pokemon. The current buddy distances are 1, 3, 5, and 20 kilometres distance.  Returns a JSON object where each key is the distance needed to gain a candy and the value is a list containing multiple objects holding distance, Pokemon ID, Pokemon name and optionally the form."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/pokemon_buddy_distances.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pokemon_candy_to_evolve(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all Pokemon that evolve there will be a specific candy requirement. This API groups the evolvable Pokemon into the specific candy requirements for each.  Returns a JSON object where each key is the amount of candy needed to evolve and the value is a list containing multiple objects holding candy needed to evolve, Pokemon ID, Pokemon name and optionally the form."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/pokemon_candy_to_evolve.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pokemon_stats(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each Pokemon has three base stats, attack, defense and stamina which determine how innately strong it is in each of these areas. These effect how much HP and damage each move can do along with its level and the typing of the moves.  Returns a JSON array where each element is a dict containing the pokemon name, ID, base stamina, base attack, and base defense."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/pokemon_stats.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nesting_pokemon(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Only specific Pokemon will nest and this API lets you get the name and ID of all Pokemon known to currently nest.  Returns a JSON dict with the keys being the Pokemon ID, the values are an array containing the pokemon name and ID."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/nesting_pokemon.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pokemon_maximum_cp(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each Pokemon has a maximum CP that it may achieve if it is a perfect pokemon (with 15 attack, stamina, and defense) and is level 40. This API returns the maximum CP for each Pokemon.  Returns a JSON array where each element is a dict containing the pokemon name, ID, and maximum CP."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/pokemon_max_cp.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shiny_pokemon(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a JSON dict with the keys being the Pokemon ID, the values are an array containing the Pokemon name, ID, and how that shiny can be found.There are four main keys determining how the shiny can be found:  found_wild - True if the shiny Pokemon is found in the wild found_raid - True if the shiny Pokemon was once in, or currently is in raids as a shiny possibility found_egg - True if the shiny Pokemon can be hatched from an egg as a shiny found_evolution - True if the Pokemon can be evolved from another shiny Pokemon"
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/shiny_pokemon.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alolan_pokemon(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In Generation 7 of the Pokemon video games Alolan Pokemon were released. These are pokemon who canonically evolved in a different area so are subtly different from the standard Pokemon. These all look different and have different typing to classical pokemon. This API allows you to get a list of currently released alolan Pokemon.  Returns a JSON dict with the keys being the Pokemon ID, the values are an array containing the pokemon name and ID."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/alolan_pokemon.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fast_moves(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each Pokemon has a fast and charged move. This API allows you to download the full list of fast moves in the current Pokemon Go game master.  Returns a JSON array where each element is a dict containing the stamina_loss_scaler, name, power, duration, energy_delta and type."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/fast_moves.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pokemon_names(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a json dict with the keys being the pokemon ID, the values are an array containing the pokemon name and ID."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/pokemon_names.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charged_moves(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each Pokemon has a fast and charged move. This API allows you to download the full list of charged moves in the current Pokemon Go game master.  Returns a JSON array where each element is a dict containing the stamina_loss_scaler, name, power, duration, critical_chance, energy_delta and type."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/charged_moves.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def released_pokemon(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Currently in Pokemon Go the majority of Pokemon have been released from the first three generations of Pokemon games. This API lets you get the full list of Pokemon that are currently released.  Returns a json dict with the keys being the pokemon ID, the values are an array containing the pokemon name and ID."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/released_pokemon.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weather_boosts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "During different weather certain types will be boosted. When they are boosted Pokemon of the boosted types will be found at a higher level, and moves of that type will be more powerful. This API lists what each weather type boosts.  Returns a JSON dict where each key is the weather type and the value is an array of boosted types."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/weather_boosts.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def possible_ditto_pokemon(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ditto is a Pokemon that can transform into any Pokemon. In the wild you find him by catching a Pokemon which then transforms into Ditto. You cant tell which Pokemon might turn into a Ditto however if someone has caught a Ditto it will be a Ditto for everyone. This API allows you to get the list of Pokemon which are potentially a Ditto.  Returns a JSON dict with the keys being the Pokemon ID, the values are an array containing the pokemon name and ID."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/possible_ditto_pokemon.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pokemon_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Pokemon have either one or two types, these types affect the strength of moves and weaknesses it has to opponents moves. This API lists the types of each Pokemon.  Returns a JSON array where each element is a dict containing type (an array of one or two items), Pokemon ID, Pokemon name and optionally the form."
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/pokemon_types.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pokemon_encounter_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When encountering a Pokemon there are a number of metrics that influence the catch rate, and what it does during the encounter. This API groups all the information that influences an encounter together.  Returns a JSON array where each element is a dict containing attack_probability, base_capture_rate, base_flee_rate, dodge_probability, max_pokemon_action_frequency, min_pokemon_action_frequency, Pokemon ID, Pokemon name and optionally the form.  The fields attack_probability, base_capture_rate, base_flee_rate, and dodge_probability are a value from 0 to 1. 0 represents 0% chance of it ocurring each turn and 1 represents 100% chance. The fields max_pokemon_action_frequency, and min_pokemon_action_frequency are in seconds representing the maximum and minimum amount of time between an action.  ["
    
    """
    url = f"https://pokemon-go1.p.rapidapi.com/pokemon_encounter_data.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

