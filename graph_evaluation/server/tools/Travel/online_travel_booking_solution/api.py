import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def david_peeter(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tripfro platform consist of many components assembled to get your one stop travel software and travel technology, to automate travel business process and configured in many ways to meet your business goals. Our service never stops with the sales of our softwares or API Solutions. It goes on in the form of after-sales service since we understand that our products are designed to satisfy your requirements for many years to come. Whatever your Business requirement, Tripfro offers completely managed best fit online travel booking solution. Join with us to expand your business with exclusive benefits.
		•	Flight
		Get lowest airfares, highest commission on domestic and international flight tickets to 5000 cities.
		•	Hotel
		Search and book the best hotel for your customers with highest commissions! . We offer you more Hotels across the world.
		•	Car
		We provide customers with international car rental and private transfer that are tailored to meet their needs.
		•	Transfers
		Your customers will be met at the Hotel and taken to hotel with variety of vehicles.
		
		THE TRIPFRO ADVANTAGE EARN MORE
		ON EVERY BOOKING
		Tripfro booking Engines designed to reduce cost and increase revenues, enable you to deliver the highest level of customer service.
		•	Trustworthy Security
		•	Support for multi-location environment
		•	Real time online bookings
		•	Flexible commission and mark-up settings against suppliers
		•	Reduced time and cost of report design and distribution
		•	Global Suppliers
		•	Empower people at all levels of your organization to make informed measures using software that is familiar and easy to use
		•	Largest Travel Network
		Our platform is powered by 70+ suppliers across flight, hotels, car,transfers and other services.
		•	Full Service Model
		For all Type of travel companies, we have several product bundles available.
		•	Increase Profitability
		Access to Tripfro will give you best Inventory and Pricing worldwide
		•	Products & Services
		Access to best rates for flights, hotels ,cars, transfers and more. World Class Support, Realtime Dashboard and Reporting.
		•	Widest Product Range
		Grow Your Travel Business & Earn in More with Tripfro
		•	Best Benefits
		Exclusive benefits like highest commission, widest range of products and best prices.
		
		
		To know more about TripFro, please visit 
		https://www.tripfro.com/"
    
    """
    url = f"https://online-travel-booking-solution.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-travel-booking-solution.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

