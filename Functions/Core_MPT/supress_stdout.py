import os, sys
from contextlib import contextmanager

@contextmanager
def supress_stdout(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """

    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            fd = sys.__stdout__.fileno()
        else:
            fd = sys.stdout.fileno()
    except:
        fd = sys.stdout.fileno()


    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        try:
            if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
                sys.__stdout__.close() # + implicit flush()
                os.dup2(to.fileno(), fd) # fd writes to 'to' file
                sys.__stdout__ = os.fdopen(fd, 'w') # Python writes to fd
            else:
                sys.stdout.close() # + implicit flush()
                os.dup2(to.fileno(), fd) # fd writes to 'to' file
                sys.stdout = os.fdopen(fd, 'w') # Python writes to fd
        except:
            sys.stdout.close()  # + implicit flush()
            os.dup2(to.fileno(), fd)  # fd writes to 'to' file
            sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
