from setuptools import setup, find_packages

setup(
    name='explorador',
    version='1.0.0',
    description='funciones útiles para explorar datos',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'statsmodels',
        'scikit-learn'
    ],
    py_modules=['script_miscelano']
)