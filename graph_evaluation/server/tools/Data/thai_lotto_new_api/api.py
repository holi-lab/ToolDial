import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def modern_internet(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ယခု Endpoint မှာတော့ နံနက် ၉ခွဲ နှင့် နေ့လည် ၂နာရီ မော်ဒန်၊ အင်တာနက် ထွက်ဂဏန်း နှင့် 12:01 မိနစ်၊ ညနေ 4:30 ထွက်မယ့် ဂဏန်းများကို တိုက်ရိုက်ပြသဖို့အတွက်ပဲဖြစ်ပါတယ်။"
    
    """
    url = f"https://thai-lotto-new-api.p.rapidapi.com/api/v1/results"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lotto-new-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def beta_calendar(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "2019 မှ ယနေ့လက်ရှိအချိန်အထိ ထွက်ရှိခဲ့သော 2D History များကို စုစည်းပေးထားခြင်းဖြစ်ပါသည်။"
    
    """
    url = f"https://thai-lotto-new-api.p.rapidapi.com/api/v1/beta-calendar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lotto-new-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2d(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "တစ်နှစ်တာအတွက် ထိုင်းအစိုးရ ထီပိတ်မည့်ရက်များကို ကြိုတင်သိရှိနိုင်မယ့် Endpoint ပဲဖြစ်ပါတယ်။"
    
    """
    url = f"https://thai-lotto-new-api.p.rapidapi.com/api/v1/holiday"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lotto-new-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ယခု နေရာကနေတစ်ဆင့် ယခင်ထွက်ရှိခဲ့တဲ့ 3D ဂဏန်းများကို ကြည့်ရှုနိုင်မှာပဲဖြစ်ပါတယ်။"
    
    """
    url = f"https://thai-lotto-new-api.p.rapidapi.com/api/v1/threed"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thai-lotto-new-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

