from setuptools import setup, find_packages

setup(
    name="falcon-trader",
    version="0.1.0",
    description="Falcon Trading Platform - Paper Trading Bot & Orchestrator",
    author="TradingAsBuddies",
    url="https://github.com/TradingAsBuddies/falcon-trader",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "falcon_trader": ["www/*.html", "orchestrator/*.yaml"],
    },
    python_requires=">=3.9",
    install_requires=[
        "falcon-core @ git+https://github.com/TradingAsBuddies/falcon-core.git",
        "backtrader>=1.9.78",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "requests>=2.32.3",
        "Flask>=3.1.2",
        "Flask-CORS>=4.0.1",
        "pytz>=2023.3",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "falcon-trader=falcon_trader.run_orchestrator:main",
            "falcon-dashboard=falcon_trader.dashboard_server:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
