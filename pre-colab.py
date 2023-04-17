# #@title 默认标题文本
# #The code below installs 3.10 (assuming you now have 3.8) and restarts environment, so you can run your cells.

# import sys #for version checker
# import os #for restart routine

# if '3.10' in sys.version:
#   print('You already have 3.10')
# else:
#   #install python 3.10 and dev utils
#   #you may not need all the dev libraries, but I haven't tested which aren't necessary.
#   !sudo apt-get update -y
#   !sudo apt-get install python3.10 python3.10-dev python3.10-distutils libpython3.10-dev 
#   !sudo apt-get install python3.10-venv binfmt-support #recommended in install logs of the command above

#   #change alternatives
#   !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
#   !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2

#   # install pip
#   !curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
#   !python3 get-pip.py --force-reinstall

#   #install colab's dependencies
#   !python3 -m pip install setuptools ipython ipython_genutils ipykernel jupyter_console prompt_toolkit httplib2 astor

#   #minor cleanup
#   !sudo apt autoremove

#   #link to the old google package
#   !ln -s /usr/local/lib/python3.8/dist-packages/google /usr/local/lib/python3.10/dist-packages/google
#   #this is just to verify if 3.10 folder was indeed created
#   !ls /usr/local/lib/python3.10/

#   #restart environment so you don't have to do it manually
#   os.kill(os.getpid(), 9)




#@title change python version
#The code below installs 3.10 (assuming you now have 3.8) and restarts environment, so you can run your cells.

import sys #for version checker
import os #for restart routine

!echo $PATH
!whereis python
!ls -al /usr/bin | grep python
!rm /usr/bin/python3
!ln -s /usr/bin/python3.8 /usr/bin/python3
!python3 -V
!curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
!python3 get-pip.py --force-reinstall
!pip show pip
!pip install ipykernel ipython -U --force-reinstall

# #install colab's dependencies
# !python3 -m pip install setuptools ipython ipython_genutils ipykernel jupyter_console prompt_toolkit httplib2 astor

# #minor cleanup
# !sudo apt autoremove

# #link to the old google package
# !ln -s /usr/local/lib/python3.9/dist-packages/google /usr/local/lib/python3.8/dist-packages/google
# #this is just to verify if 3.8 folder was indeed created
# !ls /usr/local/lib/python3.8/

# """
# 需要在cmd中使用 python xxx.py 运行，才可实现程序重启
# """

# import sys
# import os
# import time


# def main():
#     print('主程序运行...')


# def restart_program():
#     print('3s后重启程序！')
#     time.sleep(3)
#     python = sys.executable
#     os.execl(python, python, *sys.argv)


# if __name__ == '__main__':
#     main()
#     restart_program()

print(sys.version)
