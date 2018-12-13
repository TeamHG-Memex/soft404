
import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='soft404',
    version='0.2.1',
    author='Konstantin Lopuhin',
    author_email='kostia.lopuhin@gmail.com',
    description='A classifier for detecting soft 404 pages',
    license='MIT',
    url='https://github.com/TeamHG-Memex/soft404',
    packages=['soft404'],
    include_package_data=True,
    install_requires=[
        'six',
        'html_text>=0.4.1',
        'numpy',
        'scikit-learn>=0.18',
        'scipy>=0.18',
        'webstruct>=0.4',
        'parsel>=1.4',
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
