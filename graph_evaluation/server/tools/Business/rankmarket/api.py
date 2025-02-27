import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def rankmarket(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rankmarket is groupon marketplace for internet marketers since 2015.trusted by 5000+ customers. we have newest & great marketing products with cheapest.
		#rankmarketorg #groupbuyhumansynthesysstudiootos #groupbuycreativefabricaannual #groupbuypoweradspypremiumannual #GB SPYTOOLS
		Address :99 Crewes Close, London, SE1 9RA, United Kingdom
		Hotlline 0869.325.309
		Email: support@rankmarket.org
		Website https://rankmarket.org 
		https://www.rankmarket.org/product-category/gb-spytools/ 
		https://www.rankmarket.org/product/group-buy-human-synthesys-studio-otos/ 
		https://www.rankmarket.org/product/group-buy-creativefabrica-annual/ 
		https://www.rankmarket.org/product/group-buy-poweradspy-premium-annual/ 
		https://www.facebook.com/rankmarketnet/ 
		https://dribbble.com/rankmarket/about
		https://my.desktopnexus.com/rankmarket/#ProfileComments
		http://git.newslab.iith.ac.in/snippets/6630
		https://www.instapaper.com/p/rankmarket
		https://gitlab.aic.ru/snippets/4395
		https://www.reddit.com/user/rankmarket
		https://www.spreaker.com/user/15812199
		https://gitlab-test.eclipse.org/rankmarket
		https://www.scoop.it/u/rank-market
		https://rankmarket.hpage.com/
		https://comicvine.gamespot.com/profile/rankmarket/about-me/
		https://www.provenexpert.com/rankmarket/
		https://gitlab.tue.nl/rankmarket
		https://git.qt.io/rankmarket
		https://gitlab.pasteur.fr/rankmarketcom
		https://www.beatstars.com/rankmarket
		https://www.webmasterpro.de/user/rankmarket/
		https://seedandspark.com/user/rank-market
		https://www.jigsawplanet.com/rankmarket?viewas=1ab0bf1f06df
		https://coub.com/rankmarket
		https://api.rakuten.net/user/rankmarket
		https://xclams.xwiki.org/xwiki/bin/view/XWiki/rankmarket?category=profile
		http://photozou.jp/user/top/3300019
		https://www.vingle.net/posts/4166741
		https://notionpress.com/author/435193
		https://www.divephotoguide.com/user/rankmarket
		https://hypothes.is/users/rankmarket
		https://onmogul.com/rankmarket
		https://learn.acloud.guru/profile/rankmarket
		https://www.hulkshare.com/rankmarket
		https://timeswriter.com/members/rankmarket/profile/
		https://www.csslight.com/profile/rank-market
		https://git.feneas.org/rankmarket
		http://www.good-tutorials.com/users/rankmarket
		https://www.cnccode.com/user/rankmarket
		https://amara.org/en/profiles/profile/yJpthAb5GoR-OKcBUbgCyU4Dz2wvADMBny3SQurpCbc/
		https://godotengine.org/qa/user/rankmarket
		https://hypixel.net/members/rankmarket.5718319/#about
		https://bookme.name/rankmarket
		https://cliqafriq.com/rankmarket
		https://www.logobids.com/users/rankmarketcom
		https://code.getnoc.com/-/snippets/1172
		https://reedsy.com/discovery/user/rankmarket
		https://www.xosothantai.com/members/rankmarketorg.502388/"
    
    """
    url = f"https://rankmarket.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankmarket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

