from setuptools import setup, find_packages
#from Cython.Build import cythonize
import numpy as np

descrip='''
====// --------- \\\\====
Author:
Lucas Siemons
====// --------- \\\\====
'''

setup(
    name='data2ensembles',
    version='0.1',
    author='L. Siemons',
    author_email='lucas.siemons@googlemail.com',
    packages=find_packages(),
    #ext_modules=cythonize("data2ensembles/csa_spec_dens.pyx"),
    #license='LICENSE.txt',
    package_data={'data2ensembles': ['dat/*dat'],  ['dat/*txt']},
    include_package_data=True,
    description=descrip,
    include_dirs=[np.get_include()]
    #long_description=open('README.md').read(),
    #install_requires=['numpy','scipy>=0.17.0', 'matplotlib'],
)