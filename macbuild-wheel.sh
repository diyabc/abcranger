export CC=/usr/local/opt/llvm/bin/clang
export CXX=/usr/local/opt/llvm/bin/clang++
export LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
export CPPFLAGS=-I/usr/local/opt/llvm/include
export CXXFLAGS=-stdlib=libc++
export CIBW_SKIP="cp27-* cp35-* cp36-* cp37-* pp*  *-manylinux_i686 *-win32"
export CIBW_BEFORE_BUILD="pip install cmaketools cmake ninja vswhere"
export CIBW_TEST_REQUIRES="pytest h5py"
export CIBW_TEST_COMMAND="pytest -s {project}/test/test-pyabcranger.py -v"
python -m cibuildwheel --output-dir wheelhouse --platform macos