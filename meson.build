project('tpsimilarity', 'python',
  version : '0.1.0',
  license : 'MIT',
  meson_version : '>=0.56.0',
  default_options : ['warning_level=2'])

py3 = import('python').find_installation('python3')

py3.extension_module('tpsimilarity',
  'tpsimilarity/similarity.py',
  'tpsimilarity/version.py',
  install : true)

py3.install_sources('tpsimilarity/__init__.py', subdir : 'tpsimilarity')

py3.install_sources('project.toml', subdir : '.')

py3.install_sources('README.md', subdir : '.')

py3.install_sources('LICENSE', subdir : '.')

py3.install_sources('requirements.txt', subdir : '.')

py3.install_sources('MANIFEST.in', subdir : '.')

py3.install_sources('examples', subdir : 'examples')

py3.install_sources('notebooks', subdir : 'notebooks')
