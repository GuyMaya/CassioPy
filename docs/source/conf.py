# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import mock

# List of modules to mock import
MOCK_MODULES = ["scipy.stats", "sklearn"]
for mod_name in MOCK_MODULES:
   sys.modules[mod_name] = mock.Mock() 

# -- Path setup --------------------------------------------------------------
import os

sys.path.insert(0, os.path.abspath("../../cassiopy"))
# sys.path.insert(0, os.path.abspath('../../cassiopy'))


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.append(os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("../sphinxext"))
sys.path.extend(
    [
        # numpy standard doc extensions
        os.path.join(os.path.dirname(__file__), "..", "../..", "sphinxext")
    ]
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cassiopy'
copyright = '2024, Maya GUY'
author = 'Maya GUY'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx.ext.extlinks',
    'sphinx.ext.autosummary',
    'sphinx_design',
    'sphinx.ext.doctest',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

root_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# -- Options for HTMLHelp output ---------------------------------------------
html_theme_options = {
    "show_toc_level": 3,
    "navigation_depth": 3,
    "navbar_end": [ "theme-switcher", "icon-links" ],
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            "url": "https://github.com/GuyMaya/CassioPy",  # required
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
   ],
  #     "secondary_sidebar_items": {
  #   "**": ["page-toc", "sourcelink"],
  #   "index": ["page-toc"],
  # }
}