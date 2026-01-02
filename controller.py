import os
import sys
import subprocess

ENV_BIN = sys.exec_prefix
ENV_PYTHON_BA = os.path.join(os.path.split(ENV_BIN)[0], "badApple/bin/python")
ENV_PYTHON_4DSTEM = os.path.join(os.path.split(ENV_BIN)[0], "py4dstem_gui311/bin/python")
print(ENV_BIN)
print(ENV_PYTHON_BA)
print(ENV_PYTHON_4DSTEM)

SIZE = 32
#subprocess.run(["source", "activate", "root"])
#subprocess.run('bash -c "source activate root; python -V"', shell=True)
#subprocess.run("conda init", shell = True)

#for i in range(1, 6029):

for i in range(1,6029):
    #print(f"{i:0=6}")
    #subprocess.run(["conda", "activate", "badApple"])
    subprocess.run([ENV_PYTHON_BA, "downscale.py", f"{SIZE}", f"{i:0=6}"])
    subprocess.run([ENV_PYTHON_BA, "frames_to_atoms.py", f"{SIZE}", f"{i:0=6}"]) 
    subprocess.run([ENV_PYTHON_BA, "atoms_to_dp.py", f"{SIZE}", f"{i:0=6}"]) 
    #subprocess.run(["conda", "activate", "py4dstem_gui311"])
    subprocess.run(["timeout", "10", ENV_PYTHON_4DSTEM, "-m", "py4D_browser.runGUI", f"dp/frame{i:0=6}.h5"]) 
    subprocess.run(["rm", f"dp/frame{i:0=6}.h5"])
    #subprocess.run("gnome-screenshot")