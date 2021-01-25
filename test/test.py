import subprocess
import os
import glob
import time

def test_modelchoice(path):
    """Run basic Model choice example
    """
    for filePath in glob.glob('modelchoice_out.*'):
        os.remove(filePath)
    subprocess.run(path)

def test_modelchoice_multi(path):
    """Run basic multi target Model choice example
    """
    for filePath in glob.glob('modelchoice_out.*'):
        os.remove(filePath)
    subprocess.call([path,"-b","statobsRF2.txt"])

def test_estimparam(path):
    """Run basic Parameter estimation example
    """
    for filePath in glob.glob('estimparam_out.*'):
        os.remove(filePath)
    subprocess.call([path,"--parameter","ra","--chosenscen","3","--noob","50"])

def test_estimparam_multi(path):
    """Run basic multi target Parameter estimation example
    """
    for filePath in glob.glob('estimparam_out.*'):
        os.remove(filePath)
    subprocess.call([path,"-b","statobsRF2.txt","--parameter","ra","--chosenscen","3","--noob","50"])

def test_parallel(path):
    """Check multithreaded performance
    """
    for filePath in glob.glob('estimparam_out.*'):
        os.remove(filePath)
    time1 = time.time()
    subprocess.call([path,"--parameter","ra","--chosenscen","3","--noob","50","-j","1"])
    time2 = time.time()
    subprocess.call([path,"--parameter","ra","--chosenscen","3","--noob","50","-j","8"])
    time3 = time.time()
    assert (time2-time1) > 1.5 * (time3 - time2)
