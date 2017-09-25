#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Tom Augspurger'
SITENAME = 'datas-frame'
SITEURL = 'https://tomaugspurger.github.io'

PATH = 'content'

TIMEZONE = 'US/Central'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

DISPLAY_PAGES_ON_MENU = True

DEFAULT_PAGINATION = 10
THEME = "pelican-themes/pelican-hss"
CSS_FILE = "main.css"
STATIC_PATHS = [
    "images/",
    "modern_2_method_chaining_files/",
    "Indexes_files/"
    "modern_2_method_chaining_files/",
    "modern_4_perforamnace_files/",
    "modern_5_tidy_files/",
    "modern_6_visualization_files/",
    "modern_7_timeseries_files/",
]

PLUGIN_PATHS = [
    'pelican-plugins'
]
PLUGINS = [
    'pandoc_reader',  # waiting on https://github.com/liob/pandoc_reader/pull/4
    'pelican-yaml-metadata',
    # 'render_math',
]
RELATIVE_URLS = True
FEED_RSS = 'feed'

DISQUS_SITENAME = "tomaugspurger"
GOOGLE_ANALYTICS = "UA-48304175-1"
