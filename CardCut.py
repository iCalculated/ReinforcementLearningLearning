"""
Human
    Search ideal tag

    Is rep? Is sense?
    tag simplification
    "lizard in control of US gov" "democratics win midterms" "borders cause violence"

Find
    Keywords
Yank
    .txt
Reliability Assessment
    Number of articles
    Qualification

    Compare? (John Baudrillard)
"""
import sys
import urllib
from urllib.request import urlopen
from googlesearch import search
from bs4 import BeautifulSoup
import newspaper
from newspaper import Article
from newspaper import fulltext
import numpy as np

import inspect

import requests
import nltk
nltk.download('punkt')

currentFuncName = lambda n=0: sys._getframe(n + 1).f_code.co_name
cnn_paper = newspaper.build('http://cnn.com')

def troubleshoot(s="", end="\n", troubleshooting=True):
        if troubleshooting:
            print("[" + currentFuncName(1) + "]: " + s, end=end)


class CardCut:
    def cnn_extract(self, query="", num=10):
        return None

    def __init__(self, tld="co.in", num=10, stop=1, pause=2):
        self.tld = tld
        self.defNum = num
        self.stop = stop
        self.pause = pause
        self.articleUrls = None

    def find_articles(self, query=None, num=10):
        if query is None:
            query = input("Query:")
        troubleshoot("Search commencing", end="->")
        articleUrls = search(query=query, tld=self.tld, num=num, stop=self.stop, pause=self.pause)
        troubleshoot("Search complete")

        return articleUrls

    def text_strip(self, url):
        text = fulltext(requests.get(url).text)

    def article_strip(self, url):
        info = np.zeros(6, dtype='O')
        article = Article(url)
        troubleshoot("Downloading article", end="->")
        article.download()
        troubleshoot("Download complete")
        troubleshoot("Parsing article", end="->")
        article.parse()
        troubleshoot("Parsing complete")
        info[0] = url
        info[1] = article.authors
        info[2] = article.publish_date
        info[3] = article.text

        troubleshoot("Natural language processing", end="->")
        article.nlp()
        troubleshoot("Processing complete")
        info[4] = article.keywords
        info[5] = article.summary
        troubleshoot("Information returned")
        return info



    def query(self, query=None, num=10):
        if query == None:
            query = input("Query:")
        for url in search(query=query, tld=self.tld, num=num, stop=self.stop, pause=self.pause):
            html = urllib.urlopen(url).read()
            soup = BeautifulSoup(html)

            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()  # rip it out

            # get text
            text = soup.get_text()

            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)

            troubleshoot(text)


def runLoop():
    card_cutter = CardCut()
    for count, url in enumerate(card_cutter.find_articles("Public Charge", num=100)):
        data = card_cutter.article_strip(url)
        troubleshoot("Article " + str(count) + ": Write Attempted...")
        np.savetxt('test' + str(count) + '.out', data, delimiter=',', fmt='%s')
        troubleshoot("Article " + str(count) + ": Write to " + "test" + str(count) + ".out complete")


if __name__ == "__main__":
    runLoop()
