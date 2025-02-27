import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_classification_report(ft_input: int, threes_attempts_input: int, points_input: int, twos_attempts_input: int, assists_input: int, twos_input: int, fg_input: int, turnovers_input: int, minutes_input: int, classification_report: str, threes_input: int, fga_input: int, fta_input: int, usage_input: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint obtains the Classification Report from given data. In this report includes the accuracy."
    
    """
    url = f"https://will-you-make-it-to-the-nba.p.rapidapi.com/Classification_Report/{classification_report}"
    querystring = {'FT_input': ft_input, 'Threes_attempts_input': threes_attempts_input, 'points_input': points_input, 'twos_attempts_input': twos_attempts_input, 'assists_input': assists_input, 'twos_input': twos_input, 'FG_input': fg_input, 'turnovers_input': turnovers_input, 'Minutes_input': minutes_input, 'Threes_input': threes_input, 'FGA_input': fga_input, 'FTA_input': fta_input, 'usage_input': usage_input, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "will-you-make-it-to-the-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def predict(fg_input: int, threes_attempts_input: int, twos_attempts_input: int, minutes_input: int, fga_input: int, ft_input: int, points_input: int, turnovers_input: int, twos_input: int, usage_input: int, assists_input: int, threes_input: int, fta_input: int, prediction: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return whether a player will make it to the NBA based on input stats."
    
    """
    url = f"https://will-you-make-it-to-the-nba.p.rapidapi.com/nba_or_not/{Prediction}"
    querystring = {'FG_input': fg_input, 'Threes_attempts_input': threes_attempts_input, 'twos_attempts_input': twos_attempts_input, 'Minutes_input': minutes_input, 'FGA_input': fga_input, 'FT_input': ft_input, 'points_input': points_input, 'turnovers_input': turnovers_input, 'twos_input': twos_input, 'usage_input': usage_input, 'assists_input': assists_input, 'Threes_input': threes_input, 'FTA_input': fta_input, }
    if prediction:
        querystring['Prediction'] = prediction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "will-you-make-it-to-the-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

