#!/usr/bin/env python

from setuptools import setup

setup(name='rpyglmnet',
      version='0.1',
      #packages = find_packages(),
      description='glmnet wrapped in rpy',
      author='Dmytro S Lituiev',
      url='https://github.com/DSLituiev/rpyglmnet',
      packages=['.'],
      install_requires = ["rpy2>=2.7.8", "numpy>=1.10.4", "scikit-learn>=0.17", ],
      setup_requires = ["rpy2>=2.7.8", "numpy>=1.10.4", "scikit-learn>=0.17", ],
      #dependency_links = [ "https://pypi.python.org/pypi/rpy2" ],
     )

