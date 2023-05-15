import gym
import os
import numpy as np
from collections import deque
from rlpd.data.dataset import Dataset
from rlpd.data import MemoryEfficientReplayBuffer

# DATA_DIR = "/juno/group/bin_sort_data/0501_data/0501_target_forward_put_jar_from_table_to_bin.npz"
# DATA_DIR = "/juno/group/bin_sort_data/0504_data/data/0501_target_forward_put_peanut_jar_from_table_to_bin.npz"
# DATA_DIR = "/juno/u/jingyuny/projects/p_bridge/data/0501_target_forward_put_jar_from_table_to_bin.npz" # 40 jar
DATA_DIR = "/juno/u/jingyuny/projects/p_bridge/data/demos/0426_target_forward_put_pepsi_from_table_to_a_bin.npz" # 1000 pepsi
DIR_PATH = "/juno/group/bin_sort_data/0504_data/data" # 25, 40 demo diverse
def process_expert_dataset(expert_dataset):
    all_observations = {"pixels": expert_dataset['img_obs'], "state": expert_dataset['lowdim_ee']}
    all_next_observations = {"pixels": expert_dataset['next_img_obs'], 'state': expert_dataset['next_lowdim_ee']}
    return {
        "observations": all_observations,
        "next_observations": all_next_observations,
        "actions": expert_dataset['actions'],
        "rewards": expert_dataset['rewards'],
        "terminals": expert_dataset['dones'],
        "masks": 1.0 - expert_dataset['dones']
    }


class MyDataBuffer(MemoryEfficientReplayBuffer):
    def __init__(
        self,
        env: gym.Env,
        dataset_level: str,
        pixel_keys: tuple = ("pixels",),
        capacity: int = 500_000,
    ):

        super().__init__(
            env.observation_space,
            env.action_space,
            capacity=capacity,
            pixel_keys=pixel_keys,
        )
        framestack = env.observation_space[pixel_keys[0]].shape[-1]
        if DIR_PATH is not None:
            for name in os.listdir(DIR_PATH):
                data = np.load(os.path.join(DIR_PATH, name), allow_pickle=True)
                print("Loaded", name)
                dataset_dict = process_expert_dataset(data)
                images = dataset_dict["observations"]["pixels"]
                for i in range(images.shape[0]):
                    if i >= framestack:
                        next_stacked_frames.append(images[i])
                        data_dict = dict(
                            observations={"pixels": np.stack(stacked_frames, axis=-1)}, #"state": dataset_dict["observations"]["state"][i]},
                            actions=dataset_dict["actions"][i],
                            rewards=dataset_dict["rewards"][i],
                            masks=1 - np.float32(dataset_dict["terminals"][i]),
                            dones=np.float32(dataset_dict["terminals"][i]),
                            next_observations={
                                "pixels": np.stack(next_stacked_frames, axis=-1)#, "state": dataset_dict["next_observations"]["state"][i]
                            },
                        )
                        self.insert(data_dict)
                        stacked_frames.append(images[i])
                    else:
                        stacked_frames = deque(maxlen=framestack)
                        next_stacked_frames = deque(maxlen=framestack)
                        while len(stacked_frames) < framestack:
                            stacked_frames.append(images[i])
                            next_stacked_frames.append(images[i])
        else:
            data = np.load(DATA_DIR, allow_pickle=True)
            print("Loaded", DATA_DIR)
            dataset_dict = process_expert_dataset(data)
            images = dataset_dict["observations"]["pixels"]
            for i in range(images.shape[0]):
                if i >= framestack:
                    next_stacked_frames.append(images[i])
                    data_dict = dict(
                        observations={"pixels": np.stack(stacked_frames, axis=-1)}, #"state": dataset_dict["observations"]["state"][i]},
                        actions=dataset_dict["actions"][i],
                        rewards=dataset_dict["rewards"][i],
                        masks=1 - np.float32(dataset_dict["terminals"][i]),
                        dones=np.float32(dataset_dict["terminals"][i]),
                        next_observations={
                            "pixels": np.stack(next_stacked_frames, axis=-1)#, "state": dataset_dict["next_observations"]["state"][i]
                        },
                    )
                    self.insert(data_dict)
                    stacked_frames.append(images[i])
                else:
                    stacked_frames = deque(maxlen=framestack)
                    next_stacked_frames = deque(maxlen=framestack)
                    while len(stacked_frames) < framestack:
                        stacked_frames.append(images[i])
                        next_stacked_frames.append(images[i])
class MyData(Dataset):
    def __init__(
        self
    ):
        dataset_dict = np.load(DATA_DIR, allow_pickle=True)
        data = process_expert_dataset(dataset_dict)
        
        

        super().__init__(data)
