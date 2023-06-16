# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
import changepoint


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "changepoint"
copyright = "2023, Promise.ai"
author = "Promise.ai"

sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    # ----------------------
    # sphinx extensions
    # ----------------------
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    # ----------------------
    # third-party extensions
    # ----------------------
    "autodocsumm",
    "numpydoc",
    "sphinx_autosummary_accessors",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_favicon",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
version = changepoint.__version__
release = version


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
