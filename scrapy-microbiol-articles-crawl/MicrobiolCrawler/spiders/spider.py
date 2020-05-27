# -*- coding: utf-8 -*-
import scrapy
import json

# name: название статьи
# author: авторы
# year: год
# volume: том
# number: номер(рядом с томом)
# annotation: текст аннотации статьи

data = {}
data['articles'] = []

class microbiolArticlesSpider(scrapy.Spider):

    name = "Crawler"
    start_urls = [
        'https://microbiol.elpub.ru/jour/issue/archive',
    ]

    def parse(self, response):

        TOM_PAGES_SELECTOR = '#issue > h4 > a::attr(href)'
        tom_pages_urls = response.css(TOM_PAGES_SELECTOR).getall()
        print(tom_pages_urls)

        for tom_url in tom_pages_urls:
            if tom_url is not None:
                yield scrapy.Request(
                    response.urljoin(tom_url),
                    callback=self.parseTom
                )

    def parseTom(self, response):

        article_urls = response.css('#content > div.issueArticle.flex > div > div.title > a::attr(href)').getall()

        print('@')
        print(article_urls)
        print('@')

        for article_url in article_urls:
            if article_url is not None:
                yield scrapy.Request(
                    response.urljoin(article_url),
                    callback=self.parseArticle
                )

    def parseArticle(self, response):

        # article = {}

        for meta in response.css("div#content"):
            if response.css("#articleAbstract > div ::text").get is not None:

                yield {
                    # authorString > em > a:nth-child(1)
                    'name': meta.css("#articleTitle > h1::text").get(),
                    'author': ' '.join(meta.css("#authorString > em > a::text").getall()),
                    'year': getYear(response.css("#breadcrumb > a:nth-child(2) ::text").get()),
                    # breadcrumb > a:nth-child(2)
                    'volume': getTom(response.css("#breadcrumb > a:nth-child(2) ::text").get()),
                    'number': getNumber(response.css('#breadcrumb > a:nth-child(2) ::text').get()),
                    'annotation': ' '.join(meta.css("#articleAbstract > div ::text").getall())
                }

            print('================')
            print(meta.css("#articleTitle > h1::text").get())
            print(getNumber(response.css('#breadcrumb > a:nth-child(2) ::text').get()))
            print(meta.css("#articleAbstract > div ::text").get())


def getTom(strToCut):
    beginCutWithIndex = strToCut.find(' ')
    if beginCutWithIndex < 3:
        return ''
    result = strToCut[beginCutWithIndex + 1:strToCut.find(',')]
    return result


def getYear(strToCut):
    result = strToCut[strToCut.find('(') + 1:strToCut.find(')')]
    return result


def getNumber(strToCut):
    endCuttWithSymbol = strToCut.find(' ')
    if endCuttWithSymbol > 1:
        endCuttWithSymbol = strToCut.find(' ', endCuttWithSymbol + 1)
        endCuttWithSymbol = strToCut.find(' ', endCuttWithSymbol + 1)
    endCuttWithSymbol = strToCut.find(' ', endCuttWithSymbol + 1)
    result = strToCut[strToCut.find('№') + 2:strToCut.find(' ', endCuttWithSymbol)]
    return result
