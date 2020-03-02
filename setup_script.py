import os
import sys

import scipy
from cx_Freeze import setup, Executable

PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')


build_exe_options = {'packages': ['numpy'],
                     'includes': ['matplotlib.backends.backend_tkagg'],
                     'include_files': [(os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tcl86t.dll'),
                                        os.path.join('lib', 'tcl86t.dll')),
                                       (os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tk86t.dll'),
                                        os.path.join('lib', 'tk86t.dll'))
                                       # add here further files which need to be included as described in 1.
                                      ]}
base = None

if sys.platform == 'win32':
    base = None

# GUI applications require a different base on Windows (the default is for a console application).
exe = Executable(
    script="VSTS.py",
    base=base
)

var = os.environ['Path']

setup(name="VSTS2",
      version="2.1",
      description="My VSTS2 application!",
      options={"build_exe": build_exe_options},
      executables=[exe])

def find_data_file(Lx_Icons):
    if getattr(sys, 'frozen', False):
        # The application is frozen
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        datadir = os.path.dirname(__file__)
    return os.path.join(datadir, Lx_Icons)