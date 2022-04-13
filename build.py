import os
import subprocess
import sys


requirementsPath = './requirements'


def setupReq():
    print(f'Installing requirements')
    if os.path.exists(requirementsPath):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install','-r', requirementsPath])

def setUpAbslutepath():
    root = os.path.abspath('.')
    data = os.path.abspath('./Data')
    print(f'Setting up absolute path')
    try:
        with open('./CSC413.py','w') as f:
            f.write(f'CSC413_ROOT_PATH = "{root}"\n')
            f.write(f'DATA = "{data}"\n')
                
    except Exception as e:
        raise e
        print('Failed to open file ./__init__.py')
        
        
# def installMain():
#     print(f'Installing main package')
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e','.'])


def main():
    setupReq()
    print('\n')
    setUpAbslutepath()
    print('\n')
    # installMain()
       

if __name__ == "__main__":
    main()