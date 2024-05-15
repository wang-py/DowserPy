import subprocess


def prepare_receptor(input_args):
    popen_args = ("./pythonsh", "prepare_receptor4.py")
    popen_args += input_args
    print(popen_args)
    popen = subprocess.Popen(popen_args, stdout=subprocess.PIPE)
    popen.wait()
