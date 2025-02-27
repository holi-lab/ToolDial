import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def spend_bonus_category_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls a list of all spend bonus categories"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-spendbonuscategory-categorylist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def credit_card_image(cardkey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls the card image URL for single credit card"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-card-image/{cardkey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spend_bonus_category_card(categoryid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls a list of all credit cards within a spend bonus category"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-spendbonuscategory-categorycard/{categoryid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def annual_spend_bonus(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that offer an annual spend bonus"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-benefit-annualspend"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sign_up_bonus_airline_miles(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that offer airline miles as a sign-up bonus"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-signupbonus-airline"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfer_program_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls a list of all travel programs that points can be transfered to from a bank points program (e.g American Express Membership Rewards)"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-pointtransfer-transferprogramlist/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfer_program_card(transferpartnerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls a list of all credit cards that can transfer points to the transfer program (Hilton, United, etc.)"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-pointtransfer-transferprogramcard/{transferpartnerid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detail_by_card(cardkey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls detailed information for a single credit card"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-detail-bycard/{cardkey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_card_by_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search credit card names by string"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-detail-namesearch/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sign_up_bonus_annual_fee_waived(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that waive the first year annual fee"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-signupbonus-annualfeewaived"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sign_up_bonus_cash_bonus(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that offer cash as a sign-up bonus"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-signupbonus-cash"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_spend_airline(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that offer airline miles for daily spend"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-dailyspend-airline"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_spend_hotel(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that offer hotel points for daily spend"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-dailyspend-hotel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_spend_2x_cashback(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that offer 2% cashback"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-dailyspend-2cashback"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_spend_cruise(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that offer cruise miles for daily spend"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-dailyspend-cruise"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def free_checked_bag(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that offer a free checked bag as part of its benefits"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-benefit-checkedbag"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def free_hotel_night(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that offer a free hotel night on the card anniversary renewal"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-benefit-hotelnight"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trusted_traveler_credit(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that offer a Trusted Traveler credit"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-benefit-tsa"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lounge_access(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that offer aiport lounge access as part of the card benefits"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-benefit-loungeaccess"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sign_up_bonus_hotel_points(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This pulls all credit cards that offer hotel points as a sign-up bonus"
    
    """
    url = f"https://rewards-credit-card-api.p.rapidapi.com/creditcard-signupbonus-hotel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rewards-credit-card-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

