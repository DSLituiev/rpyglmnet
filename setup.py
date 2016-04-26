#!/usr/bin/env python

from distutils.core import setup

setup(name='rpyglmnet',
      version='0.1',
      #packages = find_packages(),
      description='glmnet wrapped in rpy',
      author='Dmytro S Lituiev',
      url='https://github.com/DSLituiev/rpyglmnet',
      packages=['rpyglmnet'],
      install_requires = ["rpy2>=2.7.8", "numpy>=1.10.4", "sklearn>=0.18", ],
      setup_requires = ["rpy2>=2.7.8", "numpy>=1.10.4", "sklearn>=0.18", ],
      #dependency_links = [ "https://pypi.python.org/pypi/rpy2" ],
     )

