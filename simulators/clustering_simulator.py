from obstacles.sbpd_map import SBPDMap
from trajectory.trajectory import SystemConfig, Trajectory
from simulators.failure_detection_simulator import Simulator
from simulators.clustering import Clustering
import cv2
import numpy as np

class SBPDSimulator(Simulator):
    name = 'SBPD_Simulator'

    def __init__(self, params):
        assert(params.obstacle_map_params.obstacle_map is SBPDMap)
        super(SBPDSimulator, self).__init__(params=params)
        self.i = 0
        #self.use_expert = True
        self.clustering = Clustering()
        self.pose_data = []
        self.use_clustering = True
        self.pred_hist = []

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Return the robot's observation from configuration config
        or pos_nk3.
        """
        curr_obs = self.obstacle_map.get_observation(config=config, pos_n3=pos_n3, **kwargs)
        # cv2.imwrite(str(self.i) + '.png', curr_obs[0, :, :, ::-1])       # UNCOMMENT FOR ROLLOUT
        curr_obs = cv2.resize(curr_obs[0], (int(curr_obs.shape[1]*0.21875), int(curr_obs.shape[2]*0.21875)))
        curr_obs = np.expand_dims(curr_obs, axis=0)
        # self.i += 1
        return curr_obs

    def get_observation_from_data_dict_and_model(self, data_dict, model):
        """
        Returns the robot's observation from the data inside data_dict,
        using parameters specified by the model.
        """
        if hasattr(model, 'occupancy_grid_positions_ego_1mk12'):
            kwargs = {'occupancy_grid_positions_ego_1mk12':
                      model.occupancy_grid_positions_ego_1mk12}
        else:
            kwargs = {}

        img_nmkd = self.get_observation(pos_n3=data_dict['vehicle_state_nk3'][:, 0],
                                        **kwargs)
        return img_nmkd

    def _reset_obstacle_map(self, rng):
        """
        For SBPD the obstacle map does not change
        between episodes.
        """
        return False

    def _update_fmm_map(self):
        """
        For SBPD the obstacle map does not change,
        so just update the goal position.
        """
        if hasattr(self, 'fmm_map'):
            goal_pos_n2 = self.goal_config.position_nk2()[:, 0]
            self.fmm_map.change_goal(goal_pos_n2)
        else:
            self.fmm_map = self._init_fmm_map()
        self._update_obj_fn()

    def _init_obstacle_map(self, rng):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p)

    def _render_obstacle_map(self, ax):
        p = self.params
        self.obstacle_map.render_with_obstacle_margins(ax, start_config=self.start_config,
                                                       margin0=p.avoid_obstacle_objective.obstacle_margin0,
                                                       margin1=p.avoid_obstacle_objective.obstacle_margin1)
    
    def _iterate(self, config):
        """ Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and relevant planner data"""
        
        # Get the current image and the state
        current_image_nmkd = self.get_observation(config=config, pos_n3=None)
        current_pos_n5 = config.position_heading_speed_and_angular_speed_nk5()[:, 0].numpy()
        print(" ")
        print(current_pos_n5[0])
        self.pose_data.append(current_pos_n5[0])
        np.save('pose_data.npy', self.pose_data)
        cv2.imwrite('/data/aryaman/Visual-Navigation-Release/sim_images/' + str(self.i) + '.png', current_image_nmkd[0, :, :, ::-1])
 
        # Get the prediction from the clustering algorithm
        pred = self.clustering.cluster('/data/aryaman/Visual-Navigation-Release/sim_images/', self.i)
        print('pred:', pred)

        # Update prediction history
        self.pred_hist.append(pred)
        print('pred history:', self.pred_hist)
        if len(self.pred_hist) > 3:
            self.pred_hist.pop(0)  # Keep only the last 3 predictions

        if self.use_clustering:
            # Check if the last three predictions indicate failure
            if len(self.pred_hist) == 3 and all(p != "SAFE" for p in self.pred_hist):
                print('FAILURE DETECTED')
                planner_data = self.planner_expert.optimize(config)
            else:
                if pred == 'SAFE':
                    print('SAFE DETECTED')
                    planner_data = self.planner.optimize(config)
                else:
                    print('SAFE DETECTED')
                    planner_data = self.planner_expert.optimize(config)

        else:
            planner_data = self.planner.optimize(config)

        print(" ")
        self.i += 1

        trajectory_segment, trajectory_data, commanded_actions_nkf = self._process_planner_data(config, planner_data)
        next_config = SystemConfig.init_config_from_trajectory_time_index(trajectory_segment, t=-1)
        return trajectory_segment, next_config, trajectory_data, commanded_actions_nkf
