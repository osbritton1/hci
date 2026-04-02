# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import subprocess

subprocess.call("cd ..; doxygen", shell=True)

sys.path.insert(0, os.path.abspath('../../'))

project = 'HCI for PySCF'
copyright = '2026, Simon Britton'
author = 'Simon Britton'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "breathe", "sphinx.ext.mathjax", "sphinx.ext.intersphinx"]

templates_path = ['_templates']
exclude_patterns = []
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None)
}
autodoc_mock_imports = ["pyscf"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

breathe_projects = { "hci": "../doxy_files/xml/" }
breathe_default_project = "hci"
breathe_default_members = ('members', 'protected-members', 'undoc-members')