# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
needs_sphinx = '4.3'

project = 'cassiopy'
copyright = '2024, Maya GUY'
author = 'Maya GUY'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'sphinxcontrib.bibtex',  # Ajoutez cette ligne pour utiliser bibtex

]

autosummary_generate = True
numpydoc_class_members_toctree = False


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

root_doc = 'index'

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

bibtex_bibfiles = ['Documentation/referencemixturemodel.bib', 'Documentation/referenceARI.bib', 'Documentation/referencePDF.bib',  'Documentation/referenceBIC.bib']  # Spécifiez le fichier BibTeX


# The html index document.
master_doc = 'index'

# The latex index document
latex_doc = 'index_latex'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']
html_css_files = ['custom.css']

html_js_files = [
    'custom.js',
]

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
      "secondary_sidebar_items": {
    "**": ["page-toc", "sourcelink"],
    "index": ["page-toc"],
  }
}

html_logo = "_static/Images/Cassiopy_logo.png"

html_favicon = "_static/Images/Cassiopy_favicon.ico"

