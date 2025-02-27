import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_forecast_by_lottery_draw_type(lotterydrawtype: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Şans oyunu türüne göre sıradaki çekiliş için yapay zeka tahminlerini getirir."
    
    """
    url = f"https://milli-piyango1.p.rapidapi.com/TicketForecast/GetForecastByLotteryDrawType"
    querystring = {'lotteryDrawType': lotterydrawtype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "milli-piyango1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_last_stats_by_lottery_type(lotterytype: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Şans oyunu türüne göre güncel istatistikleri getirir."
    
    """
    url = f"https://milli-piyango1.p.rapidapi.com/StatDetail/GetLastStatsByLotteryType"
    querystring = {'lotteryType': lotterytype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "milli-piyango1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lottery_draw_result(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Şans oyunlarından talep edilen ID'li çekiliş sonuç detaylarını getirir."
    
    """
    url = f"https://milli-piyango1.p.rapidapi.com/LotteryDrawResult/GetDetailById"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "milli-piyango1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_next_draws(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Şans oyunları için sıradaki çekilişlerin takvimini getirir."
    
    """
    url = f"https://milli-piyango1.p.rapidapi.com/GetNextDraws"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "milli-piyango1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_last_results(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Şans oyunları için son yapılan çekilişlerin özet bilgilerini getirir."
    
    """
    url = f"https://milli-piyango1.p.rapidapi.com/LotteryDraw/GetLastResults"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "milli-piyango1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

