from pandas.io.common import urljoin
import requests as req
from bs4 import BeautifulSoup
import pandas as pd
import re
from imdb import Cinemagoer

prefix = "https://www.imdb.com/search/title/?title_type=tv_series&start="
suffix = "&ref_=adv_nxt"
numbers = ["1", "51", "101"]
all_shows = []
links = []
rank = list(range(1,201))
for number in numbers:
  url = prefix + number + suffix
  html = req.get(url).content
  soup = BeautifulSoup(html, 'html.parser')
  titles = soup.find_all('h3', {"class": "lister-item-header"})
  for link in soup.findAll('a', attrs={'href' : re.compile("title/tt")}):
    split = link.get('href').split("/")
    if split[2][2:] not in links and split[2] != 'tt0253754' and split[2] != 'tt0092455':
      links.append(split[2][2:])
  for title in titles:
    all_shows.append(title.a.contents[0])
shows = pd.DataFrame(list(zip(all_shows, links, rank)), columns = ['name', 'ids', 'imdb_rank'])

show_list = []
ia = Cinemagoer()
for id in shows['ids']:
  show_list.append(ia.get_movie(id))
  
info = []
for show in show_list:
  info.append((show['title'], show['series years'], show['genres']))
  
infodf = pd.DataFrame(info, columns = ['name', 'years', 'genre'])

shows = shows.merge(infodf, how = 'left', on = ['name'])
shows.drop_duplicates

ratings = []
for show in show_list:
  try:
    ratings.append((show['title'], show['rating'], show['votes']))
  except:
    pass

imdb = pd.DataFrame(ratings, columns = ['name', 'imdb_rating', 'imdb_votes'])

shows = shows.merge(imdb, how = 'left', on = ['name'])
shows.drop_duplicates

shows.to_csv('raw/shows.csv',index=False)