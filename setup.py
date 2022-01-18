from setuptools import setup

setup(name='grok-jax',
      version='0.0.1',
      install_requires=[
          'jax',
          'dm-haiku',
          'wandb',
          'tqdm',
          'optax',
          'hydra-core',
          'omegaconf',
          'numpy',
          'torch',
          'sympy',
          'mod',
          'blobfile',
          'flatdict',
          ]
      )
