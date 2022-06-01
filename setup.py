from setuptools import setup, find_packages

setup(
    name='price_prediction',
    version='0.1.0',
    packages=find_packages(include=['price_prediction']),
    install_requires=[
        'coverage==6.2',
        'flake8==4.0.1',
        'Flask==2.0.2',
        'lightgbm==3.3.1',
        'numpy==1.21.5',
        'pandas==1.3.5',
        'pytest==6.2.5',
        'pytest-cov==3.0.0',
        'requests==2.26.0',
        'scikit-learn==1.0.1',
        'waitress==2.0.0',
        'yapf==0.31.0',
    ]
)
