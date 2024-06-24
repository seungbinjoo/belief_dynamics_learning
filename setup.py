from setuptools import setup, find_packages

try:
    with open('README.md') as file:
        long_description = file.read() 
except IOError:  # file not found
    pass

setup(name="belief_dynamics_learning",
      long_description=long_description,  # __doc__, # can be used in the vpsto.py file
      long_description_content_type = 'text/markdown',
      version='1.0.0',
      description="Learning belief dynamics through contacts",
      author="Lara Brudermueller",
      author_email="larab@robots.ox.ac.uk",
      maintainer="Lara Brudermueller",
      maintainer_email="larab@robots.ox.ac.uk",
      url="https://github.com/brudermueller/belief_dynamics_learning", 
      license="BSD",
      packages=find_packages(),
      install_requires=["numpy"],#, "mujoco", "robotic"],
      extras_require={
            "plotting": ["matplotlib"],
      },
)
