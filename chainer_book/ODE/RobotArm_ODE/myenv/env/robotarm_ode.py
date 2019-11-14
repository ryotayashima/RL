# -*- coding: utf-8 -*-
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

import ode

logger = logging.getLogger(__name__)

col = False

def Collision_Callback(args, geom1, geom2):
    contacts = ode.collide(geom1, geom2)
    world, contactgroup = args
    for c in contacts:
        c.setBounce(0)
        c.setMu(2)
        j = ode.ContactJoint(world, contactgroup,c)
        j.attach(geom1.getBody(), geom2.getBody())
        global col
        col = True

world = ode.World()
world.setGravity((0, 0, -9.81))

body1 = ode.Body(world)
M = ode.Mass()
M.setSphere(25.0, 0.05)
M.mass = 1.0
body1.setMass(M)

body2 = ode.Body(world)
M = ode.Mass()
M.setBox(25, 0.2, 0.5, 0.2)
M.mass = 1.0
body2.setMass(M)

space = ode.Space()
Arm_Geom = ode.GeomSphere(space, radius=0.05)
Arm_Geom.setBody(body1)
Ball_Geom = ode.GeomBox(space, (0.2, 0.5, 0.2))
Ball_Geom.setBody(body2)
Floor_Geom = ode.GeomPlane(space, (0, 0, 1), 0)
contactgroup = ode.JointGroup()

class RobotArmODEEnv(gym.Env):
    metadata = {
        'render.modes':['human', 'rgb_array'],
        'video.frames_per_second':50
    }
    def __init__(self):
        self.Col = False
        self.gravity = 9.81
        self.cartmass = 1.0
        self.cartwidth = 0.5
        self.cartheight = 0.2
        self.cartPosition = 0
        self.ballPosition = 1
        self.ballRadius = 0.05
        self.ballVelocity = 1
        self.force_mag = 10.0
        self.tau = 0.02

        self.x_threshold = 1
        self.y_threshold = 1

        high = np.array([
            self.x_threshold,
            self.y_threshold,
            self.x_threshold,
            self.y_threshold
        ])

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        if action==0:
            fx = self.force_mag
            fy = 0
        if action==1:
            fx = 0
            fy = self.force_mag
        if action==2:
            fx = -self.force_mag
            fy = 0
        if action==3:
            fx = 0
            fy = -self.force_mag

        space.collide((world, contactgroup), Collision_Callback)
        body1.setForce((fx, fy, 0))
        world.step(self.tau)
        contactgroup.empty()
        bx, by, bz = body2.getPosition()
        rx, ry, rz = body1.getPosition()
        self.state = (rx, ry, bx, by)
        done = False

        if bx > self.x_threshold or bx <-self.x_threshold \
            or by > self.y_threshold or by < -self.y_threshold \
            or rx > self.x_threshold or rx < -self.x_threshold \
            or ry > self.y_threshold or ry < -self.y_threshold:
            done = True
        reward = 0.0001
            
        if  bx*bx + by*by <= 0.04 and bx*bx + by*by > 0.01:
            
#             if rx > 0.8 or rx <-0.8 \
#             or ry > 0.8 or ry < -0.8:
#                 reward = -100.0
#             elif rx < 0.5 and rx > -0.5 and ry < 0.5 and ry > -0.5:
#                 reward = -1.0
            
            reward = 0.01/(bx*bx + by*by)
            
        if  bx*bx + by*by < 0.01:
            reward = 0.01/(bx*bx + by*by+0.001)
                
            if (bx-0.7)*(bx-0.7)+by*by < 0.25:
                reward = 0.25/((bx-0.7)*(bx-0.7) + by*by+0.01)
            
        return np.array(self.state), reward, done, {}
    def _reset(self):
        body1.setPosition((0.8, 0, 0.05))
        body1.setLinearVel((0, 0, 0))
        body1.setForce((0, 0, 0))
        body2.setPosition((0.3, 0, 0.1))
        body2.setLinearVel((0, 0, 0))
        body2.setForce((0, 0, 0))
        body2.setQuaternion((1, 0, 0, 0))
        body2.setAngularVel((1, 0, 0, 0))

        rx, ry, rz = body1.getPosition()
        bx, by, bz = body2.getPosition()
        self.state = (rx, ry, bx, by)
        self.steps_beyond_done = None
        self.by_dot = 0
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400
        world_width = self.x_threshold*2
        scale = screen_width/world_width
        cartwidth = self.cartwidth*scale
        cartheight = self.cartheight*scale


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            ball1 = rendering.make_circle(0.05*scale)
            self.balltrans1 = rendering.Transform()
            ball1.add_attr(self.balltrans1)
            ball1.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(ball1)

            l = self.cartheight/2*scale
            w = self.cartwidth/2*scale
            ball2 = rendering.FilledPolygon([(-l,w),(l,w),(l,-w),(-l,-w)])
            self.balltrans2 = rendering.Transform(translation=(0,0))
            ball2.add_attr(self.balltrans2)
            ball2.set_color(0, 0, 0)
            self.viewer.add_geom(ball2)

            ball3 = rendering.make_circle(0.1*scale)
            self.balltrans3 = rendering.Transform(translation=(screen_width/2.0, screen_height/2.0))
            ball3.add_attr(self.balltrans3)
            ball3.set_color(0.8,0.8,0.8)
            self.viewer.add_geom(ball3)

        if self.state is None: return None

        x1, y1, z1 = body1.getPosition()
        x2, y2, z2 = body2.getPosition()
        self.balltrans1.set_translation(x1*scale+screen_width/2, y1*scale+screen_height/2)
        self.balltrans2.set_translation(x2*scale+screen_width/2, y2*scale+screen_height/2)
        if body2.getRotation()[1] < 0:
            self.balltrans2.set_rotation(math.acos(body2.getRotation()[0]))
        else:
            self.balltrans2.set_rotation(3.14 - math.acos(body2.getRotation()[0]))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
