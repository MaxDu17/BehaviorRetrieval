import numpy as np
import pybullet as p


def connect_headless(gui=False):
    if gui:
        cid = p.connect(p.SHARED_MEMORY)
        if cid < 0:
            p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.resetDebugVisualizerCamera(cameraDistance=0.8,
                                 cameraYaw=180,
                                 cameraPitch=-40,
                                 cameraTargetPosition=[0.6, 0, -0.4])
    p.setRealTimeSimulation(False)
    p.stepSimulation()


def disconnect():
    p.disconnect()


def reset():
    p.setRealTimeSimulation(False)
    p.resetSimulation()


def setup_headless(timestep=1./240, solver_iterations=150, gravity=-10):
    p.setPhysicsEngineParameter(numSolverIterations=solver_iterations)
    p.setTimeStep(timestep)
    p.setGravity(0, 0, gravity)
    p.setRealTimeSimulation(False)
    p.stepSimulation()


def load_state(path):
    p.restoreState(fileName=path)
