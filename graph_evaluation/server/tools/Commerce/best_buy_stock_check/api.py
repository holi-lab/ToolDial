import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_product_info(best_buy_sku_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Below is an example response,
		
		```
		 {
		  "sku": "6462956",
		  "brand": "NVIDIA",
		  "manufacturer": {
		    "name": "D&amp;H DISTRIBUTING NVIDIA",
		    "modelNumber": "900-1G133-2518-000"
		  },
		  "primarySupplierName": "D&amp;H DISTRIBUTING NVIDIA",
		  "shortName": "NVIDIA GEFORCE RTX 3080 TI",
		  "longName": "NVIDIA - GeForce RTX 3080 Ti 12GB GDDR6X PCI Express 4.0 Graphics Card - Titanium and Black",
		  "description": "The GeForce RTX™ 3080 Ti delivers the ultra performance that gamers crave, powered by Ampere—NVIDIA’s 2nd gen RTX architecture. It’s built with enhanced RT Cores and Tensor Cores, new streaming multiprocessors, and superfast G6X memory for an amazing gaming experience.",
		  "links": {
		    "self": {
		      "href": "http://data.bestbuy.com/v1/item/id/sku/6462956"
		    },
		    "master": {
		      "href": "http://data.bestbuy.com/master/product/id/sku/6462956"
		    }
		  },
		  "images": "https://pisces.bbystatic.com/image2/BestBuy_US/images/products/6462/6462956_sd.jpg",
		  "dimensions": {
		    "unitOfMeasure": "IN",
		    "height": 2.2,
		    "width": 4.4,
		    "depth": 11.2
		  },
		  "weight": {
		    "unitOfMeasure": "LB",
		    "weight": 3
		  },
		  "shippingInfo": {
		    "shippingWeight": {
		      "unitOfMeasure": "LB",
		      "weight": 4.7
		    },
		    "singleUnitInBox": {
		      "unitOfMeasure": "IN",
		      "height": 6.6,
		      "width": 14.1
		    }
		  },
		  "url": "https://www.bestbuy.com/site/nvidia-geforce-rtx-3080-ti-12gb-gddr6x-pci-express-4-0-graphics-card-titanium-and-black/6462956.p?skuId=6462956",
		  "stock": "Sold Out",
		  "mrsp": 1,
		  "upc": "812674024851",
		  "upcs": {
		    "upc3": "812674024851",
		    "upc4": "400064629566",
		    "upc1PrimaryIndicator": false,
		    "upc2PrimaryIndicator": false,
		    "upc3PrimaryIndicator": true,
		    "upc4PrimaryIndicator": false
		  },
		  "averageCost": 919.08,
		  "unitRetail": 1199.99,
		  "countryOfOrigin": {
		    "code": "US",
		    "name": "United States"
		  },
		  "inStockDate": "2021-06-03T05:00Z",
		  "brickAndMortarStreetDate": "2021-06-03T05:00Z",
		  "storePickUp": true,
		  "skuQuantityLimit": 1
		}
		```"
    
    """
    url = f"https://best-buy-stock-check.p.rapidapi.com/productinfo/{best_buy_sku_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "best-buy-stock-check.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

