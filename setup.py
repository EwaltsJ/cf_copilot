from setuptools import find_packages, setup

setup(
    name='cf_copilot',
    version="0.0.1",
    description="Project Description",
    packages=find_packages(),
    install_requires=[],  # Use `pip install -r requirements.txt` instead
    test_suite='tests',
    include_package_data=True,
    zip_safe=False,
)
