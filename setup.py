from setuptools import setup, find_packages

setup(
    name="ddos-rl-mitigation",
    version="1.0.0",
    description="DDoS Attack Mitigation Using Deep Reinforcement Learning in SDN",
    author="Prashant Kaushik",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.65.0",
    ],
)
