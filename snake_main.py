import mujoco_py
import os
import math
xml_path = 'Model/simplified-snake.xml'
# xml_path = 'snake-copy.xml'
model = mujoco_py.load_model_from_path(xml_path)    #load xml model

# +- 90degree
#model = mujoco_py.load_model_from_xml(MODEL_XML)
sim = mujoco_py.MjSim(model)    #load simulation
viewer = mujoco_py.MjViewer(sim)

t = 0
while True:
    sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
    sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
    t += 1
    sim.step()
    print(sim.data.qpos)
    viewer.render()
    if t > 10000 and os.getenv('TESTING') is not None:
        break