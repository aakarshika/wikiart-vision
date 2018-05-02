
# Setup Instructions for Pyleargist

https://pypi.org/project/pyleargist/

## Installing FFTW

Unix Instructions: http://www.fftw.org/fftw3_doc/Installation-on-Unix.html

## Pyleargist setup instructions

1) Download the archive here: https://pypi.org/project/pyleargist/#files

2) Replace the setup.py file with this snippet below which replaces occurrences of file() with open() for Python 3 compatibility.

        from setuptools import setup
        from setuptools.extension import Extension
        from Cython.Distutils import build_ext
        import sys, os
        import numpy as np

        version = open('VERSION.txt').read().strip()

        setup(name='pyleargist',
              version=version,
              description="GIST Image descriptor for scene recognition",
              long_description=open('README.txt').read(),
              classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
              keywords=('image-processing computer-vision scene-recognition'),
              author='Olivier Grisel',
              author_email='olivier.grisel@ensta.org',
              url='http://www.bitbucket.org/ogrisel/pyleargist/src/tip/',
              license='PSL',
              package_dir={'': 'src'},
              cmdclass = {"build_ext": build_ext},
              ext_modules=[
                  Extension(
                      'leargist', [
                          'lear_gist/standalone_image.c',
                          'lear_gist/gist.c',
                          'src/leargist.pyx',
                      ],
                      libraries=['m', 'fftw3f'],
                      include_dirs=[np.get_include(), 'lear_gist',],
                      extra_compile_args=['-DUSE_GIST', '-DSTANDALONE_GIST'],
                  ),
              ],
              )

3) Run `python setup.py build`
If this gives you the error `gcc exited with status 1` scroll up to see if the `leargist.pxd` file is missing.
If it is missing in the `src` directory, download it from [Bitbucket](https://bitbucket.org/ogrisel/pyleargist/src/5d9f8ec1bb1c159ffa5b4ca1b2f6d2b303b9b871/src/?at=default)
Once you do this, run `python setup.py build` again. Now this step should succeed.

4) Run `sudo python setup.py install`.
If you get an error for Cython.Distutils not being found, check whether you're using the right version of Python using `which python`.
Alternatively, sometimes sudo picks up the wrong Python path, so give it the Python3 absolute path.
For e.g.,

`sudo /home/user/anaconda3/bin/python setup.py install`.


Even after all this runs, sometimes your Python code may not find the right Python path where leargist is installed.

In that case, copy the relevant .so file from pyleargist-2.0.5/build and copy it to your Python environment's site-packages. 
That *might* work. 


Now try to see if `leargist.color_gist()` is identifiable. Also, since the leargist library uses Pillow to read images, 
newer versions of Pillow are incompatible with pyleargist.

You want to install PIL < 3.0 - I have 2.7.0 and testing the `leargist.color_gist('ar.ppm')` was successful. 

Instead of PPM images, png images are also working fine.