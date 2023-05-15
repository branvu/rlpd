from ml_collections.config_dict import config_dict

from configs import drq_config


def get_config():
    config = drq_config.get_config()

    config.num_qs = 10
    # config.num_min_qs = 1
    config.hidden_dims = (256, 256, 256) # 3 layer MLP
    config.num_min_qs = 2 # turning on CDQ

    config.critic_layer_norm = True
    config.backup_entropy = False

    return config
