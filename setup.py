import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='soft404',
    version='0.1.2',
    author='Konstantin Lopuhin',
    author_email='kostia.lopuhin@gmail.com',
    description='A classifier for detecting soft 404 pages',
    license='MIT',
    url='https://github.com/TeamHG-Memex/soft404',
    packages=['soft404'],
    include_package_data=True,
    install_requires=[
        'six',
        'lxml',
        'numpy',
        'parsel',
        'scikit-learn',
        'scipy',
        'webstruct>=0.3',
    ],
    long_description=read('README.rst'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
