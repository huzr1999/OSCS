import os
from datetime import datetime
import yaml
from argparse import ArgumentParser


parser = ArgumentParser(description="Run Recursive KDE Probability Density Estimation Animation")
parser.add_argument('--config', type=str, default="configs/openbackdoor_final.yaml", help='Path to the configuration YAML file')
parser.add_argument('--dataset_name', type=str, help='Name of the dataset to use')
parser.add_argument('--poisoner_name', type=str, help='Name of the poisoner to use')
parser.add_argument('--defender_name', type=str, help='Name of the defender to use')
parser.add_argument('--T', type=int, help='Total number of test sequence')
parser.add_argument('--method', type=str, help='Method to use')
parser.add_argument('--not_load_scores', action='store_true', help='Whether to load existing scores')
parser.add_argument('--not_load_victim', action='store_true', help='Whether to load existing victim model')
parser.add_argument('--num_train_epochs', type=int, help='Number of epochs for backdoor training')
parser.add_argument('--score_name', type=str, help='Name of the score to use')
parser.add_argument('--attacker_name', type=str, help='Name of the attacker to use')
parser.add_argument('--model_name', type=str, help='Victim model name to use')
parser.add_argument('--results_path', type=str, default="./results", help='Path to save results')
parser.add_argument('--poison_ratio', type=float, help='Poison ratio in test set')
parser.add_argument('--alpha', type=float, help='User specified FAR level for OSCS')
args = parser.parse_args()

config_path = args.config
with open(config_path, "r", encoding="utf-8") as f:
	configs = yaml.load(f, Loader=yaml.Loader)

if args.poisoner_name is not None:
	configs["backdoor_kwargs"]["attacker"]["poisoner"]["name"] = args.poisoner_name

if args.dataset_name is not None:
	configs["backdoor_kwargs"]["poison_dataset"]["name"] = args.dataset_name
	configs["backdoor_kwargs"]["target_dataset"]["name"] = args.dataset_name

if args.defender_name is not None:
	configs["backdoor_kwargs"]["defender"]["name"] = args.defender_name

if args.method is not None:
	configs["experimental_kwargs"]["method"] = args.method

if args.T is not None:
	configs["experimental_kwargs"]["T"] = args.T

# print(args.not_load_victim, "+"*100)

if args.not_load_scores is not None:
	configs["experimental_kwargs"]["score_kwargs"]["load_scores"] = not args.not_load_scores

if args.not_load_victim is not None:
	configs["backdoor_kwargs"]["victim"]["load"] = not args.not_load_victim

if args.num_train_epochs is not None:
	configs["backdoor_kwargs"]["attacker"]["train"]["epochs"] = int(args.num_train_epochs)

if args.score_name is not None:
	configs["experimental_kwargs"]["score_kwargs"]["name"] = args.score_name

if args.attacker_name is not None:
	configs["backdoor_kwargs"]["attacker"]["name"] = args.attacker_name


if args.model_name is not None:
	configs["backdoor_kwargs"]["victim"]["model_name"] = args.model_name


if args.results_path is not None:
	configs["results_path"] = args.results_path


if args.poison_ratio is not None:
	configs["experimental_kwargs"]["poison_ratio"] = float(args.poison_ratio)

if args.alpha is not None:
	configs["experimental_kwargs"]["oscs_kwargs"]["alpha"] = float(args.alpha)

current_time = datetime.now()
timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_PATH = os.path.join(configs["results_path"], f"{timestamp_str}")

os.makedirs(RESULTS_PATH, exist_ok=True)