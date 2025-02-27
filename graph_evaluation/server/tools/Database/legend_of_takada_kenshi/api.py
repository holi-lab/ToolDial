import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def main(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "クエリパラメータは不要です.そのままアクセスしてください.
		**API認証等は特に設けておりませんが, プログラムによるAPIアクセスの場合はUser-Agentにリクエスト元のサービス名・アプリケーションを入力して下さい.**"
    
    """
    url = f"https://legend-of-takada-kenshi.p.rapidapi.com/takada.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "legend-of-takada-kenshi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

