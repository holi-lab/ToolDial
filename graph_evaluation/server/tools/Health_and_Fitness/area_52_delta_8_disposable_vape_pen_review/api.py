import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def area_52_delta_8_disposable_vape_pen_review(area52: str='Area 52', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Area 52 Delta 8 Disposable Vape Pen contains 500mg of high-grade Delta-8-THC distillate. The device has a ceramic heating element and a rubber tip mouthpiece for maximum comfort and ease of use. The pen is also infused with hemp-derived terpenes, including alpha-pinene, beta-caryophyllene, myrcene, and limonene. The cherry flavor is both pleasant and sweet, and the pen is easy to use.
		
		All [Area52 supplements](https://www.area52.com/) are tested by third parties, and the results are available on the website. Customers can contact the company to request up-to-date test results. The company has a large blog and legal section where consumers can get lots of information about delta-8-THC. However, the customer service of Area 52 is quite limited, and it is not recommended for beginners. But if you're looking for a great product, you're on the right page.
		
		Area52 offers a range of products, from tinctures to edibles, vapes, and tinctures. They use organically grown hemp grown in Washington and Oregon. The Delta-8 THC products are CO2-extracted and contain cannabis-derived terpene blends. These products are safe and effective, and Area 52 makes sure to follow their standards. They also have a great reputation for their products, so it's important to know how much you can safely take.
		
		If you're looking for a high-quality product, Area 52 is a solid choice. The brand's tinctures are tested by third-party labs and have received numerous awards. Many customers are satisfied with the results, but not everyone experiences the same level of effects. The prices are reasonable and there are even discounts for bulk purchases. You should take note of any fake reviews, as they're worthless.
		
		There are other options for Delta-8-THC products, but you should be aware of the risks associated with CBD, especially if you're not sure what the effects of the CBD oils are. The quality and consistency of these CBD-based products are the main factors to consider when purchasing them. It's not uncommon for a company to provide several products, so you might want to explore the options available and find the best product for your needs.
		
		Aside from a vape pen, Area 52's other Delta-8 products are disposable and compatible with viscous extracts. The disposable vape pen contains 500 mg of delta-8 THC and is great for beginners. Several reviews have praised the quality of the products. While it's still early to tell whether Area52's delta-8-THC products are safe, it's important to note that the brand has been around for a while and has been trusted for quality.
		
		The Area52 Delta-8 THC gummies come in three flavors. Berry Gelato is an indica vape that delivers fruity notes and is the ideal choice for evening use. For daytime use, Pineapple Express is the ideal choice with its sweet flavors. The Granddaddy Purple is a hybrid that brings the best of both worlds. One of the major benefits of the berry gelato is that it has an indica-derived flavor."
    
    """
    url = f"https://area-52-delta-8-disposable-vape-pen-review.p.rapidapi.com/users/nataliahampton21"
    querystring = {}
    if area52:
        querystring['Area52'] = area52
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "area-52-delta-8-disposable-vape-pen-review.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

