import argparse
import subprocess

from randomized_hyperparam_search import RandomizedSearch, ga_reader_parameter_space

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_times', help='Number of times to choose random parameters and train the model with them.',
                        type=int, default=10)
    search_args = parser.parse_args()

    random_search = RandomizedSearch(ga_reader_parameter_space)
    for n in range(search_args.n_times):
        print("Run {}...".format(n))
        params = random_search.sample()
        print(params)
        command = " ".join(["THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32 python3", "run.py"] +
                           ["--dataset clicr --mode 1 --ent_setup no-ent"] +
                           ["--{} {}".format(k, v) for k, v in params.items()]
                           )
        subprocess.call(command, shell=True)
