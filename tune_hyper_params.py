import itertools
import sys
import time
import Updated_CartPole as highway
from iteration_utilities import grouper
from multiprocessing import Process, Queue, Lock

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return [iter(iterable)]*n

def train_and_put_performance_in_queue(args, config, queue):
    performance = highway.train(args)
    queue.put((performance, config))  # Put a tuple of (performance, config) into the queue

def save_best_configuration(queue, best_config_file, lock, save_freq = 3600):
    best_performance = float('inf')
    best_config = None
    while True:
        time.sleep(save_freq)
        with lock:
            while not queue.empty():
                performance, config = queue.get()
                if performance < best_performance and performance != float(0):
                    best_performance = performance
                    best_config = config
            if best_config:
                with open(best_config_file, 'w') as f:
                    for param, value in zip(["K_epochs", "eps_clip", "gamma", "max_ep_len", "lr_actor", "lr_critic", "masking_threshold", "shield_lr", "safety_threshold"], best_config):
                        f.write(f"{param}: {value}\n")
                    f.write(f"Best Performance: {best_performance}\n")

def tune_hyper_parameters(number_of_processes = 5):
    K_epochss = [10, 50, 100, 200]
    eps_clips = [0.1, 0.2, 0.5]
    gammas = [0.8, 0.95, 0.99]
    max_ep_lens = [500, 5000]
    lr_actors = [1e-4, 5e-4, 1e-3, 5e-3]
    lr_critics = [1e-4, 5e-4, 1e-3, 5e-3]
    max_training_timesteps = int(500000)  # Adjusted for a potentially longer training duration
    masking_thresholds = [0, 50000, 100000]
    shield_lrs = [1e-4, 5e-4, 1e-3, 5e-3]
    safety_thresholds = [0.3, 0.5, 0.8]
    k_last_states = 1
    # NO GEN FOR NOW
    gen_masking_tresh = 3000000
    update_gen_timestep = 3000000
    seed = 0
    algo = 'ShieldPPO'
    # performance_queue - for determining the best paramaeters (that minimizes the shield loss function)
    best_config_file = "best_params/best_params.log"
    save_lock = Lock()
    performance_queue = Queue()
    configs = grouper(itertools.product(K_epochss, eps_clips, gammas,max_ep_lens, lr_actors, lr_critics, masking_thresholds, shield_lrs, safety_thresholds),number_of_processes)
    # Start the process to save the best configuration periodically
    save_process = Process(target=save_best_configuration, args=(performance_queue, best_config_file, save_lock))
    save_process.start()
    for n_configs in configs:
        processes = []
        for config in n_configs:
            K_epochs, eps_clip, gamma, max_ep_len, lr_actor, lr_critic, masking_threshold, shield_lr, safety_treshold = config
            args = f"--algo {algo} --seed {seed} --max_training_timesteps {max_training_timesteps} --lr_critic {lr_critic} --lr_actor {lr_actor} --gamma {gamma} --eps_clip {eps_clip} --K_epochs {K_epochs} --masking_threshold {masking_threshold} --shield_lr {shield_lr} --safety_treshold {safety_treshold} --max_ep_len {max_ep_len} --k_last_states {k_last_states} --gen_masking_tresh {gen_masking_tresh} --update_gen_timestep {update_gen_timestep}"
            p = Process(target=train_and_put_performance_in_queue, args=(args.split(), config, performance_queue))
            processes.append(p)
            p.start()
            time.sleep(2)
        for p in processes:
            p.join()
    # Signal the save process to stop
    save_process.terminate()
    save_process.join()

    #     sys.argv = args.split()
    #     highway.train()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        tune_hyper_parameters(int(sys.argv[1]))
    else:
        tune_hyper_parameters()