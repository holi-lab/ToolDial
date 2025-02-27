import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_username_mobile_legends(userid: str='838384475', zoneid: str='12322', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "insert ID and Zone ID Game Mobile Legends in Required Parameter"
    userid: Insert USER ID Mobile legend
        zoneid: insert ZONE ID Mobile Legends
        
    """
    url = f"https://check-id-game.p.rapidapi.com/api/rapid_api/cek_game_ml/{userid}/{zoneid}"
    querystring = {}
    if userid:
        querystring['UserID'] = userid
    if zoneid:
        querystring['ZoneID'] = zoneid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-game.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_username_call_of_duty_mobile(id_game: str='8370310025568788107', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "insert ID Game Call of Duty Mobile in Required Parameter"
    id_game: Insert ID GAME C.O.D MOBILE
        
    """
    url = f"https://check-id-game.p.rapidapi.com/api/rapid_api/cek_game_cod/{id_game}"
    querystring = {}
    if id_game:
        querystring['id_game'] = id_game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-game.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_username_genshin_impact(id_game: str='831826798', zone: str='asia', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "insert ID Genshin Impact in Required Parameter
		and change ur zone to ur server game, pls choice one.  default test is asia :
		asia
		usa
		europe
		china"
    id_game: Insert ID GAME Genshin Impact
        zone: choice zone server
        
    """
    url = f"https://check-id-game.p.rapidapi.com/api/rapid_api/test_game_genshin/{id_game}/{zone}"
    querystring = {}
    if id_game:
        querystring['id_game'] = id_game
    if zone:
        querystring['zone'] = zone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-game.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_username_arena_of_valor_a_o_v(id_game: str='1784903180879005', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "insert ID Arena Of Valor (A.O.V) in Required Parameter"
    id_game: Insert ID Game A.O.V
        
    """
    url = f"https://check-id-game.p.rapidapi.com/api/rapid_api/cek_game_aov/{id_game}"
    querystring = {}
    if id_game:
        querystring['id_game'] = id_game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-game.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_username_garena_free_fire(id_game: str='4012975855', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "insert ID Garena Free Fire in Required Parameter"
    id_game: Insert ID Game Free Fire
        
    """
    url = f"https://check-id-game.p.rapidapi.com/api/rapid_api/cek_game_ff/{id_game}"
    querystring = {}
    if id_game:
        querystring['id_game'] = id_game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-game.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

