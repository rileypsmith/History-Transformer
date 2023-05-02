"""
A web scraper that will get history of countries from Wikipedia for testing
out NLP methods.

@author: rileypsmith
Created: 04/25/2023
"""
from pathlib import Path

import re
import string
import time

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm


def fix_punctuation(text):
    """Replace punctuation with tokens. Remove others."""
    mapping = {
        '.': ' <.>',
        ',': ' <,>',
        '!': ' <.>',
        '<': '<',
        '>': '>',
        '-': ' '
    }
    for punc in string.punctuation:
        if not punc in mapping:
            mapping[punc] = ''
    for find, replace in mapping.items():
        text = text.replace(find, replace)
    # Convert to all lowercase as well
    text = text.lower()
    return text

def fix_contractions(text):
    """Replace common contractions with spelled out versions."""
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"don\'t", "do not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"couldn\'t", "could not", text)
    text = re.sub(r"shouldn\'t", "should not", text)
    return text
    
def remove_subtext(text):
    """Remove citations and subtext (parentheticals)."""
    text = re.sub(r'\[[0-9]+\]', '', text)
    text = re.sub(r'\([^)]*\)', '', text)
    return text

def fix_numbers(text):
    """Remove commas from numbers"""
    text = re.sub(r'(\d+?)–(\d+?)', r'\1 to \2', text)
    return re.sub(r'(\d+?),(\d+?)', r'\1\2', text)

def preprocess(text):
    """Apply all preprocessing to a tring of text."""
    text = remove_subtext(text)
    text = fix_numbers(text)
    text = fix_contractions(text)
    text = fix_punctuation(text)
    # And fix multiple spaces
    text = re.sub(r' +', ' ', text)
    return text

def load_page(url):
    """Request the given URL and parse it into a BeautifulSoup object"""
    response = requests.get(url)
    return BeautifulSoup(response.content, 'html.parser')

def parse_pagelist():
    """
    From Wikipedia, get a list of countries for which a history page exists.
    These will be pages like 'History of Turkey', for example.
    """
    url1 = 'https://en.wikipedia.org/wiki/Category:History_by_country'
    url2 = 'https://en.wikipedia.org/w/index.php?title=Category:History_by_country&subcatfrom=Somalia%0AHistory+of+Somalia#mw-subcategories'
    all_names = []
    for url in [url1, url2]:    
        soup = load_page(url)
        divs = soup.find_all('div', {'class': 'mw-category-group'})
        links = sum([[element for element in div.findChildren('a', recursive=True)] for div in divs], [])
        page_names = [link['title'].split(':')[1] for link in links if 'history of' in link['title'].lower()]
        all_names += page_names
        time.sleep(4)
    
    # Convert page names back to URLS
    base_url = 'https://en.wikipedia.org/wiki/{}'
    urls = [base_url.format(name.replace(' ', '_')) for name in all_names]
    return urls

def parse_countrylist():
    """
    From Wikipedia, get a list of all the articles about countries.
    """
    url = 'https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Countries/Popular_pages'
    soup = load_page(url)
    content_div = soup.find(id='mw-content-text')
    table = content_div.findChildren('table', {'class': 'wikitable'})[0].findChildren('tbody')[0]
    rows = list(table.findChildren('tr'))
    found_urls = []
    for row in rows[1:178]:
        link = row.findChildren('a', recursive=True)[0]
        found_urls.append('https://en.wikipedia.org' + link['href'])
    return found_urls


def extract_text(page):
    """
    Take a BeautifulSoup object and extract text in blocks. Basically, separate
    out each section and return a list of text from each section.
    """
    main_div = page.find(id='mw-content-text')
    main_div = main_div.findChildren()[0]
    children = main_div.findChildren()
    
    paragraphs = []
    started = False
    for element in children:
        if element.name == 'h2':
            started = True
            paragraphs.append([])
        elif (element.name == 'p') and started:
            paragraphs[-1].append(element.text)
            
    # Concatenate subparagraphs into one string each
    paragraphs = [[p.strip() for p in paragraph] for paragraph in paragraphs]
    paragraphs = [' <P> '.join(p) for p in paragraphs if p]
    # Preprocess the text
    paragraphs = [preprocess(p) for p in paragraphs]
    paragraphs = [p for p in paragraphs if len(p.split(' ')) >= 50]
    return paragraphs
    
def get_history(url):
    """Given a URL for a history page from wikipedia, scrape it"""
    # Try to load the webpage
    try:
        response = load_page(url)
    except:
        # No page found for this country
        return None
    return extract_text(response)

def make_history_dataset(outdir):
    """
    Go through the list of Wikipedia pages, scrape the history for each country,
    and then write it to a text file.
    """
    urls = parse_pagelist()
    time.sleep(4)
    
    # Scrape each one
    for url in tqdm(urls):
        # Get the name of this page and make a directory for it
        pagename = url.split('History_of_')[1]
        country_dir = Path(outdir, pagename)
        country_dir.mkdir()
        # Scrape the history
        history = get_history(url)
        # Replace country name throughout
        history = [p.replace(pagename, '<country>') for p in history]
        # Sleep so you don't get banned
        time.sleep(4)
        
        # If you got something, save it in paragraphs
        if history is not None:
            for i, paragraph in enumerate(history):
                outfile = str(Path(country_dir, f'paragraph_{i:03}.txt'))
                with open(outfile, 'w+') as fp:
                    fp.write(paragraph)
        
    return

def make_country_dataset(outdir):
    """
    Go through a list of Wikipedia's general country articles, scrape each one,
    and write it to text files.
    """
    urls = parse_countrylist()
    time.sleep(4)

    # Scrape each one
    for url in tqdm(urls):
        # Get the name of this page and make a directory for it
        pagename = url.split('wiki/')[1]
        country_dir = Path(outdir, pagename)
        country_dir.mkdir()
        # Scrape the history
        history = get_history(url)
        # Replace country name throughout
        history = [p.replace(pagename, '<country>') for p in history]
        # Sleep so you don't get banned
        time.sleep(4)
        
        # If you got something, save it in paragraphs
        if history is not None:
            for i, paragraph in enumerate(history):
                outfile = str(Path(country_dir, f'paragraph_{i:03}.txt'))
                with open(outfile, 'w+') as fp:
                    fp.write(paragraph)

    
