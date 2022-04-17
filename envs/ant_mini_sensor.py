import numpy as np
import math
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
from math import pi

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class AntMiniSensorEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="ant_mini_sensor_max.xml",
        target_velocity=0.5,
        sensor_level=6,
        forward_weight=4.0,
        drift_weight=1.0,
        ctrl_cost_weight=0.25,
        orientation_weight=2.0,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 0.6),        
        contact_force_range=(-1.0, 1.0),
        touch_sensor_range=(0, 4.0),
        joint_force_range=(-10.0, 10.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
    ):
        utils.EzPickle.__init__(**locals())

        self.xy_pos = np.zeros(2)

        self._target_velocity = target_velocity
        self._forward_weight = forward_weight
        self._drift_weight = drift_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._orientation_weight = orientation_weight
        self._forward_orientation = np.array([1., 0., 0., 0.])

        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range
        self._joint_force_range = joint_force_range
        self._touch_sensor_range = touch_sensor_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)
        
        self._sensor_level = int(sensor_level)
 
        self._pos_convert = np.array([[-45, 45],
                                      [-80, -30],
                                      [-45, 45],
                                      [30, 80],
                                      [-45, 45],
                                      [-80, -30],
                                      [-45, 45],
                                      [30, 80]])

         
        cwd = os.path.abspath(os.getcwd())
        xml_path = os.path.join(cwd, os.path.join('envs/assets', xml_file))

        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
    
    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )
    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def orientation_cost(self, quat):
        orientation_cost = self._orientation_weight * np.sum(np.abs(quat - self._forward_orientation))
        return orientation_cost

    def contact_cost(self, raw_contact_forces):
        #print(contact_forces)
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        return contact_cost

    def joint_force_cost(self, raw_joint_forces):
        #print(raw_joint_forces)
        min_value, max_value = self._joint_force_range
        joint_forces = np.clip(raw_joint_forces, min_value, max_value)/10.0
        joint_force_cost = self._joint_force_weight * np.sum(np.square(joint_forces))
        return joint_force_cost

    def touch_cost(self, raw_touch_values):
        min_value, max_value = self._touch_sensor_range
        touch_forces = np.clip(raw_touch_values, min_value, max_value)/4.0
        touch_cost = self._touch_sensor_weight * np.sum(np.square(touch_forces))
        return touch_cost
    
    @property
    def is_healthy(self):
        z = 0.4
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z <= z <= max_z
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def step(self, action):
        #xy_position_before = self.get_body_com("torso")[:2].copy()
        xy_position_before = np.copy(self.xy_pos)
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        xy_position_after = np.copy(self.xy_pos)
        #xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
	
        ctrl_cost = self.control_cost(action)
        orientation_cost = self.orientation_cost(observation[:4])
        #contact_cost = self.contact_cost
        contact_cost = 0

        forward_reward = self._forward_weight * abs(x_velocity - self._target_velocity)     
        drift_cost = self._drift_weight * abs(y_velocity)

        rewards = 3.0 #+ healthy_reward
        costs = forward_reward + drift_cost + ctrl_cost + orientation_cost

        reward = rewards - costs
        done = self.done

        quat_square_error = np.sum(np.square(self._forward_orientation - observation[:4]))
        
        info = {
                "reward" : reward,
                "forward_reward": forward_reward,
                "reward_ctrl": ctrl_cost,
                "reward_drift": drift_cost,
                "reward_balance": orientation_cost,
                "x_position": xy_position_after[0],
                "y_position": xy_position_after[1],
                "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
                "x_velocity": x_velocity,
                "y_velocity": y_velocity,
                "quat_square_error": quat_square_error,
               }

        return observation, reward, done, info

    def _get_obs(self):
        sensorData = self.sim.data.sensordata

        self.xy_pos = np.copy(sensorData[:2])
        #print(sensorData[:3])
        if self._exclude_current_positions_from_observation:
            sensorData = sensorData[3:]
        #print(max(sensorData))
        
        sensorData[4:12] = 2*(sensorData[4:12] - self._pos_convert[:,0])/(self._pos_convert[:,1] - self._pos_convert[:,0]) - 1 # redfine motor position [-1, 1]
        sensorData[12:20] = sensorData[12:20]/(360) # rev/sec

        sensorData[28:32] = np.clip(sensorData[28:32], 0, 1)

        if self._sensor_level == 0:
            sensorData = sensorData[:12]
        if self._sensor_level == 1:
            sensorData = sensorData[:20]
        if self._sensor_level == 2:
            sensorData = sensorData[:28]
        if self._sensor_level == 3:
            sensorData = sensorData[:32]
        if self._sensor_level == 4:
            sensorData = sensorData[:59]
        if self._sensor_level == 5:
            sensorData = sensorData[:86]

        return sensorData

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
