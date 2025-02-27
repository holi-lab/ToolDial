import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def knowband_prestashop_multi_vendor_marketplace_addon(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[PrestaShop Multi-Seller Marketplace Addon](https://www.knowband.com/prestashop-marketplace) by Knowband enable the store admin to transform their e-store into an online shopping mall. Using the Marketplace Addon, the store owner can allow multiple sellers to register and sell their goods at the Marketplace. Sellers registered at the Multi-Vendor Marketplace can update their profile, seller products, social media links and shipping services from the seller dashboards.
		
		Store admin can charge the seller commission and category commission on the sellers for selling goods at the Marketplace. PrestaShop Marketplace Addon allows the admin to charge global as well as the individual amount of commission on sellers. Using the PrestaShop Marketplace Addon, the store admin can also enable the membership plan service on their store. The subscription Service plan will decide the validity and limit of products sellers can list.
		
		PrestaShop Marketplace Addon offers individual admin and seller dashboards. Store admin can track the seller list, seller products, orders, shipping and reviews from the admin panel of [PrestaShop Marketplace ](http://www.knowband.com/blog/user-manual/prestashop-marketplace/)Addon. Sellers orders get listed in the backend of the PrestaShop Multi-Seller Marketplace. All the seller's orders and reviews are monitored and managed by the store admin.
		
		The main features of the PrestaShop Marketplace Addon are:
		1.Easy Installation and Configuration
		2.3 steps installation
		3.Membership plan Service
		4.Commission Management
		5.Order Monitoring and Management
		6.Seller Reviews and Management
		7.Seller Order handling
		8.Easy Earning Management
		9.Email Templates for Management
		10.Effortless Seller Management
		
		In case of any doubt or confusion, feel free to contact us at support@Knowband.com"
    
    """
    url = f"https://knowbands-prestashop-multi-vendor-marketplace-addon.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "knowbands-prestashop-multi-vendor-marketplace-addon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

