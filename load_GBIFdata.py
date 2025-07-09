# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 21:08:39 2023

Downloads images of insect species from GBIF

@author: Kim Bjerge
"""
import os
import requests
import warnings
import contextlib
from PIL import Image
from pygbif import occurrences
from pygbif import species
from urllib3.exceptions import InsecureRequestWarning

old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass
            
def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        print('Image with alpha channel convert to RGB')
        return im.convert('RGB')
    else:
        return im

def downloadImages(dataPath, speciesName):
    
    key = species.name_backbone(name=speciesName, rank='species')['usageKey']
    data = occurrences.search(taxonKey=key, limit=900, offfset=0) # continent='europa'
    
    count = 0
    for item in data['results']:
        #print(item['order'], item['family'], item['species'])
        imgPath = dataPath + item['order'] + ' ' + item['family'] + ' ' + item['species']
        if not os.path.exists(imgPath):
            os.mkdir(imgPath)
        media = item['media']
        if len(media) > 0:
            for mediaItem in media:
                url = ""
                if 'type' in mediaItem:
                    if mediaItem['type'] == 'StillImage':
                        url = mediaItem['identifier']
                else:
                    if 'references' in mediaItem:
                        url = mediaItem['references']
                if len(url) > 0 and url.find('naturgucker.net') == -1: # naturegucker.net insecure and not working
                    count += 1
                    print(url, count)
                    
                    #with no_ssl_verification():
                    im = Image.open(requests.get(url, stream=True).raw)
                    im = remove_transparency(im)
                    im.save(imgPath + '/' + str(count) + '.jpg')
                        
    print(speciesName, count)
    
#%% MAIN
if __name__=='__main__':

    #dataPath = 'D:/GBIF/dataset/'
    dataPath = 'C:/IHAK/hornet/dataset1/'
    
    """
    listSpecies = ['Agrotis puta',
                   'Amphipyra pyramidea',
                   'Autographa gamma',
                   'Hoplodrina ambigua',
                   'Hoplodrina blanda',
                   'Hoplodrina octogenaria', # - NB
                   'Mythimna pallens',
                   'Noctua fimbriata',
                   'Noctua pronuba',
                   'Xestia c-nigrum'
                  ]    
    """
    
    listSpecies = ['Apis melifera',
                   'Bombus lapidarius',
                   'Bombus terretris',
                   'Vespula vulgaris',
                   'Vespula germanica'
                  ]    
      

    for speciesName in listSpecies:
        downloadImages(dataPath, speciesName)
        