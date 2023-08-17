import sys
# only for server
import os
parent_path = os.path.dirname(sys.path[0])
sys.path.append(parent_path)
print(sys.path)

import numpy as np
import tensorflow as tf
from config import get_config
from trainer import Trainer
from utils import prepare_dirs, save_config, \
    prepare_config_date, save_results, evaluate_result
config = None

def main(_):
    # total number of running per fold
    no = 2
    folds = 5
    final_mklr = 0.0
    final_flr = 0.0
    final_mape = 0.0

    for ds_ind in range(folds):
        # cross validation with 5 folds
        config.ds_ind = ds_ind

        log_rsl = open("output.csv", "a")

        for i in range(no):
            prepare_dirs(config)
            prepare_config_date(config, config.ds_ind)
            # Random seed settings
            rng = np.random.RandomState(config.random_seed)
            tf.set_random_seed(config.random_seed)


            if 'mtr' in config.server_name:
                config.hist_range = list(range(60, 150, 10))
                config.sample_rate = 30
            else:
                config.hist_range = list(range(0, 41, 5))
                config.sample_rate = 15

            log_rsl.write(str(config)+'\n')
            log_rsl.flush()

            trainer = Trainer(config, rng)
            save_config(config.model_dir, config)
            config.load_path = config.model_dir

            _,HA,_,ave_train_time,_ = trainer.train(save=True)

            if config.target == 'hist':
                result_dict,mklr,flr,ave_test_time = trainer.test(HA)
                final_mklr += mklr
                final_flr += flr
                log_rsl.write('f %d,n %d,rm:%f,mklr:%f,flr:%f,mode:%s\n' %(config.ds_ind,i,config.data_rm, mklr, flr, config.mode))
                log_rsl.flush()

            tf.compat.v1.reset_default_graph()

    if config.target == 'hist':
        final_mklr = final_mklr / (no * folds)
        final_flr = final_flr / (no * folds)
        log_rsl.write('fold:%d,no.:%d,rm:%f,mklr_ave_10:%f,flr_ave_10:%f,mode:%s\n' % (folds,no,config.data_rm, final_mklr, final_flr, config.mode))

    else:
        final_mape = final_mape / (no * folds)
        log_rsl.write('fold:%d,no.:%d,rm:%f,mape_ave_10:%f,mode:%s\n' % (folds,no,config.data_rm, final_mape, config.mode))
    log_rsl.flush()


if __name__ == "__main__":
    config, unparsed = get_config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
