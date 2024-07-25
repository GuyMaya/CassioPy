# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
# import mock

# # List of modules to mock import
# autodoc_mock_imports = ["scipy.stats", "sklearn", 'scipy', 'matplotlib', 'numpy']

# -- Path setup --------------------------------------------------------------
import os

sys.path.insert(0, os.path.abspath("../cassiopy"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
needs_sphinx = '4.3'

project = 'cassiopy'
copyright = '2024, Maya GUY'
author = 'Maya GUY'
release = 'alpha'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
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
    # "show_toc_level": 3,
    # "navigation_depth": 3,
    # "navbar_end": [ "theme-switcher", "icon-links" ],
#     "icon_links": [
#         {
#             # Label for this link
#             "name": "GitHub",
#             "url": "https://github.com/GuyMaya/CassioPy",  # required
#             "icon": "fa-brands fa-square-github",
#             "type": "fontawesome",
#         }
#    ],
  #     "secondary_sidebar_items": {
  #   "**": ["page-toc", "sourcelink"],
  #   "index": ["page-toc"],
  # }
}