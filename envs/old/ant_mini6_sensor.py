import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from math import pi

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class AntMiniEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="ant_mini6.xml",
        forward_weight=5.0,
        ctrl_cost_weight=0.5,
        contact_cost_weight=0.05,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 0.6),
        contact_force_range=(0, 4.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
    ):
        utils.EzPickle.__init__(**locals())

        self.xy_pos = np.zeros(2)
        self._forward_weight = forward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)

        #self.last_action = np.zeros()

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
    
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

    def contact_cost(self, raw_contact_forces):
        #print(contact_forces)
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)/4.0
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        return contact_cost
    
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
        contact_cost = self.contact_cost(observation[-6:])
        #print(contact_cost)

        forward_reward = self._forward_weight * x_velocity - y_velocity
        #healthy_reward = self.healthy_reward
        healthy_reward = 0

        rewards = x_velocity + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        
        info = {
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()
        #print(np.concatenate((position, velocity, contact_force)))

        sensorData = self.sim.data.sensordata
        # convert joint velocity from radians to revolutions
        #sensorData[13:(13+8)] = sensorData[13:(13+8)] / (2*pi)
        #print(sensorData)
        self.xy_pos = np.copy(sensorData[:2])
        #print(sensorData[:3])
        if self._exclude_current_positions_from_observation:
            sensorData = sensorData[3:]
        #print(max(sensorData))
        
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
