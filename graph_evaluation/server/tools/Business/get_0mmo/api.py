import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_0mmo(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "0MMO is the number 1 site offering the very best software designed for Internet Marketers from 2008!
		Address Novokosinskaya 14k7, Moscow,119034, Russia
		Hotline 0945273782
		Email: support@0mmo.net
		https://www.0mmo.net 
		https://www.0mmo.net/gb 
		https://www.0mmo.net/im-tools 
		https://www.0mmo.net/wp 
		https://www.0mmo.net/tutorials 
		https://www.liveinternet.ru/users/0mmo/profile
		https://dribbble.com/0mmo/about
		https://infogram.com/0mmonet-1h7g6k0e17ry02o?live
		https://www.babelcube.com/user/0m-mo
		https://www.spreaker.com/user/15816794
		https://sketchfab.com/0mmo
		https://gitlab-test.eclipse.org/0mmo
		https://w3techs.com/users/profile/2751735
		https://www.scoop.it/topic/0m-mo
		https://0mmo.hpage.com/
		https://www.diigo.com/user/ommonet
		https://forums.iis.net/members/0mmo.aspx
		https://www.podomatic.com/podcasts/0mmonet9
		https://pinshape.com/users/1775513-0mmonet#designs-tab-open
		https://osf.io/xmzqj/
		https://gitlab.tue.nl/0mmo
		https://git.qt.io/0mmo
		https://communities.bentley.com/members/53bef9c4_2d00_8015_2d00_43ab_2d00_8a9c_2d00_d7d0b2a8d418
		https://myanimelist.net/profile/0mmo
		https://app.glosbe.com/profile/6875826798379142906
		https://8tracks.com/users/0mmo
		https://0mmo.cgsociety.org/profile
		https://api.rakuten.net/user/0mmonet
		https://www.forexfactory.com/0mmonet
		https://os.mbed.com/users/0mmonet/
		https://xclams.xwiki.org/xwiki/bin/view/XWiki/0mmonet
		http://photozou.jp/user/top/3300078
		https://www.designspiration.com/0mmonet9/saves/
		https://piqs.de/user/0mmo/account/
		https://hypothes.is/users/0mmo
		https://www.renderosity.com/users/0mmonet
		https://onmogul.com/0mmo
		https://www.hulkshare.com/0mmo
		https://writeablog.net/qur56j124j
		https://www.noteflight.com/profile/137b8a32fba12c039d5987842f5b20676af05440
		https://blogfreely.net/me/posts/
		https://www.cakeresume.com/dashboard
		https://appsliced.co/u/0mmo
		https://dashburst.com/0mmo
		https://timeswriter.com/members/0mmo/profile/
		https://www.csslight.com/profile/0mmo
		https://git.feneas.org/0mmonet9
		https://www.phuot.vn/members/0mmo.267461/
		https://startupmatcher.com/p/0mmo
		https://getpocket.com/@0mmo
		https://www.free-ebooks.net/profile/1356925/0mmo
		https://www.cnccode.com/user/0mmo
		https://amara.org/en/profiles/profile/pOFx88VGGOI-X9mHsomnquPySAFAWbW6JlrFYtYIgKU/
		https://en.eyeka.com/u/0mmo
		http://gitlab.aic.ru/0mmo
		http://www.asmetalwork.1gb.ua/forum/user/edit/56363.page
		https://band.us/band/86095136
		https://www.tripline.net/0mmo/
		https://bookme.name/0mmonet
		https://able2know.org/user/0mmo/
		https://profile.hatena.ne.jp/huyss/profile
		https://lackky.com/0mmonet
		https://www.webmasterpro.de/user/0mmonet/
		https://openuserjs.org/users/0mmo
		https://www.linkedin.com/in/0mmo/"
    
    """
    url = f"https://0mmo.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "0mmo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

