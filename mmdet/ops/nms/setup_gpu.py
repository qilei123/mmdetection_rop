from setuptools import setup, Extension
import os,re,sys,subprocess,copy
import os.path as osp
from os.path import join as pjoin
import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# 1 cuda config

nvcc_bin = 'nvcc.exe'
lib_dir = 'lib/x64'
def find_in_path(name, path):
    "Find a file in a search path"
    # Adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None
def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDA_PATH' in os.environ:
        home = os.environ['CUDA_PATH']
        print("home = %s\n" % home)
        nvcc = pjoin(home, 'bin', nvcc_bin)
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path(nvcc_bin, os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDA_PATH')
        home = os.path.dirname(os.path.dirname(nvcc))
        print("home = %s, nvcc = %s\n" % (home, nvcc))


    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, lib_dir)}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()
def _find_cuda_home():
    '''Finds the CUDA install path.'''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        if sys.platform == 'win32':
            cuda_home = glob.glob(
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
        else:
            cuda_home = '/usr/local/cuda'
        if not os.path.exists(cuda_home):
            # Guess #3
            try:
                which = 'where' if sys.platform == 'win32' else 'which'
                nvcc = subprocess.check_output(
                    [which, 'nvcc']).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
            except Exception:
                cuda_home = None

    return cuda_home
CUDA_HOME = _find_cuda_home()
def _join_cuda_home(*paths):
    '''
    Joins paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.

    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    '''
    if CUDA_HOME is None:
        raise EnvironmentError('CUDA_HOME environment variable is not set. '
                               'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)
def _is_cuda_file(path):
    return os.path.splitext(path)[1] in ['.cu', '.cuh']
# 2. extension config
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

# extensions
ext_args = dict(
    include_dirs=[numpy_include,CUDA['include']],
    library_dirs = [CUDA['lib64']],
    libraries = ['cudart'],
    language='c++',
    extra_compile_args={'cxx':["-DMS_WIN64","-MD" ], "nvcc":["-O2"]},
)

ext_modules = [
    # unix _compile: obj, src, ext, cc_args, extra_postargs, pp_opts
    Extension(
        "gpu_nms",
        sources=['gpu_nms.pyx', 'nms_kernel.cu'], #
        **ext_args
    ),
    ]
#customize_compiler_for_nvcc
def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to cc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    print('self.compiler_type ',self._compile)
    self.src_extensions+= ['.cu', '.cuh']
    self._cpp_extensions += ['.cu', '.cuh']
    original_compile = self.compile
    original_spawn = self.spawn
    # save references to the default compiler_so and _comple methods
    #default_compiler_so = self.compiler_so
    super = self._compile
    def win_wrap_compile(sources,
                        output_dir=None,
                        macros=None,
                        include_dirs=None,
                        debug=0,
                        extra_preargs=None,
                        extra_postargs=None,
                        depends=None):

        self.cflags = copy.deepcopy(extra_postargs)
        #print(self.cflags)
        extra_postargs = None

        def spawn(cmd):
            orig_cmd = cmd
            # Using regex to match src, obj and include files

            src_regex = re.compile('/T(p|c)(.*)')
            src_list = [
                m.group(2) for m in (src_regex.match(elem) for elem in cmd)
                if m
            ]

            obj_regex = re.compile('/Fo(.*)')
            obj_list = [
                m.group(1) for m in (obj_regex.match(elem) for elem in cmd)
                if m
            ]

            include_regex = re.compile(r'((\-|\/)I.*)')
            include_list = [
                m.group(1)
                for m in (include_regex.match(elem) for elem in cmd) if m
            ]

            if len(src_list) >= 1 and len(obj_list) >= 1:
                src = src_list[0]
                obj = obj_list[0]
                if _is_cuda_file(src):
                    print('compile cuda file--------------------------------')
                    nvcc = _join_cuda_home('bin', 'nvcc')
                    if isinstance(self.cflags, dict):
                        cflags = self.cflags['nvcc']
                    elif isinstance(self.cflags, list):
                        cflags = self.cflags
                    else:
                        cflags = []
                    cmd = [
                        nvcc, '-c', src, '-o', obj, '-Xcompiler',
                        '/wd4819', '-Xcompiler', '/MD'
                    ] + include_list + cflags
                    #print(cmd)
                elif isinstance(self.cflags, dict):
                    print('compile cpp file--------------------------------')
                    cflags = self.cflags['cxx']
                    cmd += cflags
                elif isinstance(self.cflags, list):
                    cflags = self.cflags
                    cmd += cflags

            return original_spawn(cmd)

        try:
            self.spawn = spawn
            return original_compile(sources, output_dir, macros,
                                    include_dirs, debug, extra_preargs,
                                    extra_postargs, depends)
        finally:
            self.spawn = original_spawn
    if self.compiler_type == 'msvc':
        self.compile = win_wrap_compile
        print('c===========================================')
        print(self.compile)
# run the customize_compiler
class custom_build_ext(build_ext):

    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)
setup(
    name='nms',
    ext_modules=cythonize(ext_modules),
    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext},
)