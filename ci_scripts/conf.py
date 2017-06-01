# -*- coding: utf-8 -*-

import sys
import os

sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.abspath('.'))

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = u'pyexample'
copyright = u'2015, rtfd'
author = u'rtfd'
version = '0.1'
release = '0.1'
language = None
pygments_style = 'sphinx'
todo_include_todos = False
html_theme = 'alabaster'
html_static_path = ['_static']
htmlhelp_basename = 'pyexampledoc'



extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.extlinks',
              'extension']

napoleon_use_admonition_for_examples = True 

exclude_patterns = [
  '_build',
  '*tests.rst'
]

source_prefix = 'bindings/python'


# = 'https://github.com/Microsoft/CNTK/blob/'

# sphinx.ext.extlinks options
extlinks = {
    'cntk': (source_prefix + '/%s', ''),
    'cntktut': (source_prefix + '/Tutorials/%s.ipynb', ''),
    'cntkwiki': ('https://docs.microsoft.com/en-us/cognitive-toolkit/%s', 'CNTK Doc - ')
}
