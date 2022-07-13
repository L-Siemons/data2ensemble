from setuptools import setup, find_packages

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
    #license='LICENSE.txt',
    package_data={'': ['data2ensembles/dat/*']},
    include_package_data=True,
    description=descrip,
    #long_description=open('README.md').read(),
    #install_requires=['numpy','scipy>=0.17.0', 'matplotlib'],
)