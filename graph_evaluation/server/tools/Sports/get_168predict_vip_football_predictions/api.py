import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def today_sure_vip_under_3_5_goals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers daily sure and precise under 3.5 goal forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/today/under3.5?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tomorrow_sure_vip_under_3_5_goals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers tomorrow sure and precise under 3.5 goal forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/tomorrow/under3.5?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def today_super_sure_vip_ht_ft_forcast(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This group delivers daily HT/FT match forecasts. All of the forecasts have been carefully selected. Visit [**168predict**](www.168predict.site) for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/today/ht-ft?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tomorrow_super_sure_vip_double_chance_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers tomorrow super sure and precise Double Chance Matches. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/tomorrow/double-chance?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yesterday_super_sure_vip_double_chance_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers yesterday Double Chance Matches. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/yesterday/double-chance?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def today_super_sure_vip_double_chance_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers daily super sure and precise Double Chance matches. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/today/double-chance?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tomorrow_super_sure_vip_draw_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers tomorrow super sure and precise **Draw** matches. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/tomorrow/draw?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yesterday_super_sure_vip_draw_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers yesterday super sure and precise **Draw** matches. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/yesterday/draw?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def today_super_sure_vip_draw_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers daily super sure and precise Draw matches. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/today/draw?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tomorrow_super_sure_vip_ht_ft_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers tomorrow super sure and precise HT/FT matches. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/tomorrow/ht-ft?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yesterday_super_sure_vip_ht_ft_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers yesterday super sure VIP HT/FT matches. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/yesterday/ht-ft?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tomorrow_super_sure_vip_prediction(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers tomorrow super sure and precise match forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/tomorrow/tip?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yesterday_super_sure_vip_prediction(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers yesterday super sure and precise match forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/yesterday?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def today_super_sure_vip_predictions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers daily super sure and precise match forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/today?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def today_sure_vip_matche_s_goals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers daily sure and precise matches for today. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/today/games?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tomorrow_sure_vip_btts_goal_goal(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers tomorrow sure and precise BTTS/Goal Goal forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/tomorrow/btts-gg?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yesterday_sure_vip_btts_goal_goal(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers yesterday BTTS/Goal Goal forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/yesterday/btts-gg?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def today_sure_vip_btts_goal_goal(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers daily sure and precise BTTS/Goal Goal forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/today/btts-gg?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yesterday_sure_vip_over_2_5_goals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers yesterday over 2.5 goal forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/yesterday/over2.5?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tomorrow_sure_vip_over_2_5_goals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers tomorrow sure and precise over 2.5 goal forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/tomorrow/over2.5?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def today_sure_vip_over_2_5_goals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers daily sure and precise over 2.5 goal forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/today/over2.5?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tomorrow_sure_vip_over_1_5_goals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers tomorrow sure and precise over 1.5 goal forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/tomorrow/over1.5?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yesterday_sure_vip_over_1_5_goals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers yesterday over 1.5 goal forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/yesterday/over1.5?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def today_sure_vip_over_1_5_goals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers daily sure and precise over 1.5 goal forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/today/over1.5?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yesterday_sure_vip_under_3_5_goals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpiont delivers yesterday under 3.5 goal forecasts. All of the forecasts have been carefully selected. Visit www.168predict.site for more information."
    
    """
    url = f"https://168predict-vip-football-predictions.p.rapidapi.com/vip/yesterday/under3.5?games_api_key=jbhdjfbghjbsjdfnkjdshgfuahoifhoi28u38y572yu8tqjewdoomcgddfdgdfgdgfgfddfswqdgkfjhgfcgcfxfxdsersrjgfdd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "168predict-vip-football-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

