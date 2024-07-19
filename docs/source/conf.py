# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#path-setup

import os
import sys
sys.path.insert(0, os.path.abspath('../../cassiopy'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cassiopy'
copyright = '2024, Maya GUY'
author = 'Maya GUY'

# The full version, including alpha/beta/rc tags
release = "alpha"
html_title = "cassiopy"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']


# -- Options for HTMLHelp output ---------------------------------------------
html_theme_options = {
    "repository_url": "https://github.com/GuyMaya/CassioPy",
    "use_repository_button": True,
    'path_to_docs': 'docs/',
    "use_download_button": False,
    "show_toc_level": 1,
    "toc_depth": 2,
    "toc_title": "Table of Contents",
}

html_sidebars = {
    "**": [
        "sbt-sidebar-nav.html",
    ]
}