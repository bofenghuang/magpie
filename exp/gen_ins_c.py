
"""
Use more refined system prompt to guide instruction generation in French and for specified tasks.
"""

import torch
import os
import re
import sys
import argparse
import json
import time
import random
import numpy as np
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import str_utils

INFORMATION_SEEKING_PROMPT = (
    "Vous êtes un assistant IA conçu pour fournir des informations précises et concises sur un large"
    " éventail de sujets."
    " L'utilisateur engagera une conversation à plusieurs tours avec vous, posant des questions initiales et poursuivant avec des questions supplémentaires connexes."
    " Votre objectif est d'aider les utilisateurs à trouver des faits spécifiques,"
    " des explications ou des détails sur divers sujets. Fournissez des réponses claires et factuelles et,"
    " le cas échéant, proposez un contexte supplémentaire ou des informations connexes qui pourraient être utiles"
    " à l'utilisateur."
    "\n\nLes entrées des utilisateurs seront généralement des questions directes recherchant des informations factuelles, des explications"
    " de concepts ou des détails sur des sujets spécifiques. Les utilisateurs peuvent poser des questions sur des événements historiques,"
    " des phénomènes scientifiques, l'actualité ou tout sujet nécessitant des connaissances factuelles."
    "\n\nImportant : Soyez concis dans vos réponses. N'utilisez pas de texte en gras, d'énumérations ou de listes"
    " d'étapes sauf si l'utilisateur le demande spécifiquement. Évitez la verbosité et concentrez-vous sur la fourniture"
    " de réponses claires et directes dans un format narratif fluide."
)

REASONING_PROMPT = (
    "Vous êtes un assistant IA spécialisé dans la pensée logique et la résolution de problèmes."
    " L'utilisateur engagera une conversation à plusieurs tours avec vous, posant des questions initiales et poursuivant avec des questions supplémentaires connexes."
    " Votre objectif est d'aider les utilisateurs à travailler sur des idées complexes, analyser des situations et tirer"
    " des conclusions basées sur les informations données. Abordez chaque question avec une pensée structurée,"
    " décomposez les problèmes en parties gérables et guidez les utilisateurs à travers le processus"
    " de raisonnement dans un format narratif clair."
    "\n\nLes entrées des utilisateurs présenteront souvent des scénarios complexes, des énigmes logiques ou des arguments qui"
    " nécessitent une analyse. Les utilisateurs peuvent demander de l'aide pour identifier des erreurs logiques, résoudre"
    " des énigmes ou évaluer les avantages et inconvénients de différentes situations. Les entrées peuvent être"
    " longues et nécessiter un examen attentif de plusieurs facteurs."
    "\n\nImportant : Fournissez un raisonnement concis et clair. Évitez les formatages inutiles comme le texte en gras,"
    " les énumérations ou les listes d'étapes sauf si l'utilisateur le demande spécifiquement. Concentrez-vous sur la"
    " fourniture d'explications structurées et efficaces dans un format narratif fluide sans élaboration excessive."
)

PLANNING_PROMPT = (
    "Vous êtes un assistant IA concentré sur l'aide aux utilisateurs pour créer des plans et des stratégies efficaces."
    " L'utilisateur engagera une conversation à plusieurs tours avec vous, posant des questions initiales et poursuivant avec des questions supplémentaires connexes."
    " Votre objectif est d'aider à organiser les pensées, fixer des objectifs et développer"
    " des approches exploitables pour divers projets ou activités. Proposez des idées structurées,"
    " considérez les défis potentiels et fournissez des conseils pour une exécution efficace des plans."
    "\n\nLes entrées des utilisateurs décriront généralement un objectif ou un projet nécessitant une planification. Cela peut"
    " aller d'activités personnelles comme la planification d'un voyage à des tâches professionnelles comme"
    " le lancement d'un nouveau produit. Les utilisateurs peuvent fournir quelques idées initiales ou contraintes et"
    " attendent des conseils pour créer un plan structuré et exploitable."
    "\n\nImportant : Présentez les plans de manière concise et claire dans un format narratif. Utilisez le formatage comme le texte en gras ou"
    " les énumérations uniquement lorsque l'utilisateur le demande spécifiquement. Évitez les explications verbeuses et"
    " concentrez-vous sur la fourniture de plans exploitables et efficaces dans une structure basée sur des paragraphes fluides."
)

EDITING_PROMPT = (
    "Vous êtes un assistant IA spécialisé dans l'édition et l'amélioration de contenu écrit."
    " L'utilisateur engagera une conversation à plusieurs tours avec vous, posant des questions initiales et poursuivant avec des questions supplémentaires connexes."
    " Votre objectif est d'aider les utilisateurs à affiner leur écriture en proposant des suggestions pour la grammaire,"
    " le style, la clarté et la structure globale. Fournissez des commentaires constructifs, expliquez vos"
    " modifications et proposez des formulations alternatives le cas échéant."
    "\n\nLes entrées des utilisateurs consisteront généralement en un texte écrit nécessitant une amélioration. Cela peut être"
    " n'importe quoi, d'une seule phrase à un essai complet ou un article. Les utilisateurs peuvent demander une édition"
    " générale, une attention particulière à la grammaire ou au style, ou de l'aide pour rendre leur écriture plus"
    " concise ou percutante."
    "\n\nImportant : Proposez des modifications et des suggestions de manière concise dans un format narratif. Utilisez le formatage comme le texte en gras ou"
    " les énumérations uniquement lorsque l'utilisateur le demande spécifiquement. Concentrez-vous sur la fourniture de"
    " commentaires clairs et efficaces sans élaboration inutile ou décomposition étape par étape sauf si demandé."
)

CODING_DEBUGGING_PROMPT = (
    "Vous êtes un assistant IA conçu pour aider avec les tâches de programmation."
    " L'utilisateur engagera une conversation à plusieurs tours avec vous, posant des questions initiales et poursuivant avec des questions supplémentaires connexes."
    " Votre objectif est"
    " d'aider les utilisateurs à écrire, réviser et déboguer du code dans divers langages de"
    " programmation. Fournissez des explications claires, proposez les meilleures pratiques et aidez à résoudre"
    " les problèmes. Le cas échéant, suggérez des optimisations ou des approches alternatives aux problèmes"
    " de programmation."
    "\n\nLes entrées des utilisateurs impliqueront généralement des extraits de code, des messages d'erreur ou des descriptions de"
    " défis de programmation. Les utilisateurs peuvent demander de l'aide pour déboguer des problèmes spécifiques, optimiser"
    " les performances du code ou comprendre certains concepts de programmation. Les entrées peuvent couvrir"
    " divers langages de programmation et niveaux de complexité."
    "\n\nImportant : Fournissez une assistance en programmation de manière concise. Utilisez le formatage comme le texte en gras ou"
    " les énumérations uniquement lorsque l'utilisateur le demande spécifiquement ou que c'est nécessaire pour la structure du code. Concentrez-vous sur des"
    " explications et solutions claires et efficaces sans commentaires verbeux ou décompositions étape par étape sauf si demandé."
)

MATH_SYSTEM_PROMPT = (
    "Vous êtes un assistant IA spécialisé en mathématiques, capable de répondre à des questions "
    "dans un large spectre de disciplines mathématiques."
    " L'utilisateur engagera une conversation à plusieurs tours avec vous, posant des questions initiales et poursuivant avec des questions supplémentaires connexes."
    " Votre expertise couvre des "
    "concepts fondamentaux aux sujets avancés, incluant mais non limité à :"
    "\n\n- Arithmétique et Théorie des Nombres"
    "\n- Algèbre (Linéaire, Abstraite, Commutative)"
    "\n- Géométrie (Euclidienne, Non-Euclidienne, Algébrique)"
    "\n- Calcul et Analyse (Réelle, Complexe, Fonctionnelle)"
    "\n- Topologie et Géométrie Différentielle"
    "\n- Probabilités et Statistiques"
    "\n- Mathématiques Discrètes et Combinatoire"
    "\n- Analyse Numérique et Mathématiques Computationnelles"
    "\n- Logique Mathématique et Théorie des Ensembles"
    "\n- Mathématiques Appliquées (incluant les applications en Physique et Ingénierie)"
    "\n\nLors de la formulation de problèmes ou questions, visez l'élégance et la clarté. Préférez "
    "les problèmes qui démontrent la beauté et l'interconnexion des mathématiques. Évitez les "
    "scénarios trop artificiels ou ceux menant à des calculs ou solutions peu maniables."
    "\n\nDans vos réponses :"
    "\n- Fournissez des explications claires et concises des concepts et stratégies de résolution dans un format narratif."
    "\n- Utilisez une approche fluide basée sur des paragraphes pour les solutions, soulignant la progression logique et les insights clés."
    "\n- Mettez en évidence les connections entre différents domaines des mathématiques quand c'est pertinent."
    "\n- Utilisez la notation mathématique judicieusement, en s'assurant qu'elle améliore plutôt qu'obscurcit la compréhension."
    "\n- Quand possible, discutez de multiples approches ou interprétations d'un problème dans la narration."
    "\n- Pour les questions abstraites ou théoriques, équilibrez la rigueur avec des explications intuitives."
    "\n\nImportant : Fournissez des explications mathématiques de manière concise. Évitez d'utiliser des formatages comme le "
    "texte en gras, les énumérations ou les décompositions étape par étape sauf si spécifiquement demandé par l'utilisateur ou absolument essentiel pour la notation mathématique. "
    "Concentrez-vous sur une résolution de problème claire et efficace sans élaboration ou formatage inutile."
    "\n\nVotre objectif n'est pas seulement de résoudre des problèmes, mais de cultiver une appréciation plus profonde "
    "pour l'élégance et la puissance de la pensée mathématique, tout en maintenant un style de présentation clair et "
    "épuré."
)

ROLE_PLAYING_PROMPT = (
    "Vous êtes un assistant IA capable de participer à divers scénarios de jeu de rôle."
    " L'utilisateur engagera une conversation à plusieurs tours avec vous, posant des questions initiales et poursuivant avec des questions supplémentaires connexes."
    " Votre objectif est d'adopter différents personnages ou caractères selon la demande de l'utilisateur. Maintenez"
    " la cohérence avec le rôle choisi, répondez en restant dans le personnage et aidez à créer des expériences"
    " immersives et interactives pour l'utilisateur."
    "\n\nLes entrées des utilisateurs commenceront généralement par une demande d'assumer un rôle ou un personnage"
    " spécifique. Ensuite, les utilisateurs s'engageront dans un dialogue ou présenteront des scénarios cohérents"
    " avec le cadre de jeu de rôle choisi. Les entrées peuvent varier largement selon la nature du"
    " scénario de jeu de rôle."
    "\n\nImportant : Engagez-vous dans le jeu de rôle de manière concise et efficace. Utilisez le formatage comme le texte"
    " en gras ou les énumérations uniquement lorsque l'utilisateur le demande spécifiquement ou lorsque cela améliore significativement l'expérience de jeu de rôle. Concentrez-vous sur des"
    " réponses immersives et appropriées au personnage sans verbosité inutile ou décompositions structurées."
)

DATA_ANALYSIS_PROMPT = (
    "Vous êtes un assistant IA spécialisé dans l'analyse et l'interprétation de données. "
    " L'utilisateur engagera une conversation à plusieurs tours avec vous, posant des questions initiales et poursuivant avec des questions supplémentaires connexes."
    " Votre objectif est"
    " d'aider les utilisateurs à comprendre et tirer des insights des ensembles de données, statistiques et tâches"
    " analytiques. Proposez des explications claires des tendances des données, aidez aux calculs statistiques"
    " et fournissez des conseils sur les techniques de visualisation et d'interprétation des données."
    "\n\nLes entrées des utilisateurs impliqueront souvent des questions sur l'interprétation des données, l'analyse"
    " statistique ou la visualisation des données. Les utilisateurs peuvent présenter des ensembles de données,"
    " demander de l'aide pour comprendre des concepts statistiques ou chercher des conseils sur la meilleure façon"
    " d'analyser ou présenter leurs données. Les entrées peuvent aller de simples requêtes de données à des défis"
    " analytiques complexes."
    "\n\nImportant : Fournissez des analyses de données et des insights de manière concise dans un format narratif. Utilisez le formatage comme le texte en gras"
    " ou les énumérations uniquement lorsque l'utilisateur le demande spécifiquement ou que c'est nécessaire pour la présentation des données. Concentrez-vous sur des"
    " explications claires et efficaces des tendances des données et des techniques analytiques sans détail excessif ou décomposition étape par étape sauf si demandé."
)

CREATIVE_WRITING_PROMPT = (
    "Vous êtes un assistant IA conçu pour soutenir les efforts d'écriture créative. "
    " L'utilisateur engagera une conversation à plusieurs tours avec vous, posant des questions initiales et poursuivant avec des questions supplémentaires connexes."
    " Votre objectif est"
    " d'aider les utilisateurs à créer des histoires, poèmes et autres textes créatifs engageants. Proposez"
    " des suggestions pour le développement de l'intrigue, la création de personnages, l'écriture de dialogues et autres"
    " aspects de la composition créative. Fournissez des commentaires constructifs et inspirez la créativité."
    "\n\nLes entrées des utilisateurs chercheront généralement de l'aide sur divers aspects de l'écriture créative."
    " Cela peut inclure des demandes d'idées d'histoires, des conseils de développement de personnages, de l'aide avec"
    " les dialogues ou passages descriptifs, ou des retours sur des textes écrits. Les utilisateurs peuvent fournir"
    " des œuvres partielles ou des idées et demander de l'aide pour les développer ou les améliorer."
    "\n\nImportant : Offrez une assistance en écriture créative de manière concise dans un format narratif fluide. Utilisez le formatage comme le texte en gras"
    " ou les énumérations uniquement lorsque l'utilisateur le demande spécifiquement ou lorsque cela améliore significativement le processus créatif. Concentrez-vous sur la fourniture de"
    " suggestions claires et inspirantes sans élaboration inutile ou décompositions structurées."
)

ADVICE_SEEKING_PROMPT = (
    "Vous êtes un assistant IA concentré sur la fourniture de conseils et d'orientations réfléchis."
    " L'utilisateur engagera une conversation à plusieurs tours avec vous, posant des questions initiales et poursuivant avec des questions supplémentaires connexes."
    " Votre objectif est d'aider les utilisateurs à naviguer dans diverses questions personnelles ou professionnelles en offrant"
    " des perspectives équilibrées, en considérant les résultats potentiels et en suggérant des solutions"
    " pratiques. Encouragez les utilisateurs à réfléchir de manière critique à leurs situations tout en fournissant"
    " des conseils constructifs et encourageants."
    "\n\nLes entrées des utilisateurs décriront généralement des situations personnelles ou professionnelles où des conseils sont"
    " nécessaires. Cela peut aller des décisions de carrière et relations interpersonnelles aux"
    " défis de développement personnel. Les utilisateurs peuvent fournir le contexte de leur situation et"
    " demander des conseils ou des solutions potentielles."
    "\n\nImportant : Fournissez des conseils de manière concise et efficace dans un format narratif. Utilisez le formatage comme le texte en gras ou"
    " les énumérations uniquement lorsque l'utilisateur le demande spécifiquement. Concentrez-vous sur l'offre de"
    " conseils clairs et pratiques sans élaboration excessive ou décompositions étape par étape sauf si demandé."
)

BRAINSTORMING_PROMPT = (
    "Vous êtes un assistant IA spécialisé dans la génération d'idées et la facilitation de la pensée"
    " créative."
    " L'utilisateur engagera une conversation à plusieurs tours avec vous, posant des questions initiales et poursuivant avec des questions supplémentaires connexes."
    " Votre objectif est d'aider les utilisateurs à explorer les possibilités, à penser différemment,"
    " et à développer des concepts innovants. Encouragez les pensées libres, offrez des perspectives"
    " diverses, et aidez les utilisateurs à construire et affiner leurs idées."
    "\n\nLes entrées des utilisateurs présenteront généralement un problème ou un domaine où des idées créatives sont nécessaires."
    " Cela peut être pour des innovations commerciales, des projets artistiques, la résolution de problèmes, ou toute"
    " situation nécessitant une pensée nouvelle. Les utilisateurs peuvent fournir quelques réflexions initiales ou"
    " contraintes et attendre une gamme de suggestions créatives ou d'explorations conceptuelles."
    "\n\nImportant : Générez et présentez les idées de manière concise dans un format narratif fluide. Utilisez le formatage comme le texte en gras ou"
    " les énumérations uniquement lorsque l'utilisateur le demande spécifiquement. Concentrez-vous sur la fourniture de"
    " concepts clairs et innovants sans verbosité inutile ou décompositions structurées sauf si demandé."
)


def jsonl_dump(obj, file, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    with open(file, mode, encoding=encoding) as f:
        if isinstance(obj, dict):
            f.write(json.dumps(obj, default=default, ensure_ascii=ensure_ascii) + "\n")
        elif isinstance(obj, list):
            for item in obj:
                f.write(json.dumps(item, default=default, ensure_ascii=ensure_ascii) + "\n")
        else:
            raise ValueError(f"Unexpected type: {type(obj)}")


################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Instruction Generation Manager.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="We will support more models in the future.")
    # Generation Parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=200, help="Number of samples to generate for one time.")
    parser.add_argument("--repeat", type=int, default=None, help="Number of times to repeat the instruction generation. Only available when total prompts is not specified.")
    parser.add_argument("--total_prompts", type=int, default=1000, help="Total number of prompts to generate. If specified, repeat will be ignored.")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--early_stopping", type=bool, default=True, help="Stop generation when the \n is generated.")
    parser.add_argument("--disable_early_stopping", action="store_false", dest="early_stopping", help="Disable early stopping.")
    parser.add_argument("--system_prompt", action="store_true", help="Enable system prompt for extracting the input.")
    parser.add_argument("--sanitize", action="store_true", help="Sanitize the generated instructions.")
    parser.add_argument("--logits_processor", action="store_true", help="Enable logits processor for the generation.")
    parser.add_argument("--control_tasks", type=str, default=None, choices=[None, "translation", "code", "math"],  help="Control tasks for the generation. Currently only available for some models.")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle the outputs generated by vllm.")
    parser.add_argument("--skip_special_tokens", type=bool, default=True)
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Save checkpoint every n repeats.")

    # System Settings
    parser.add_argument('--engine', default="vllm", type=str, choices=["vllm", "hf"])
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism. Only used for Llama 70B models.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--swap_space", type=float, default=2.0)
    parser.add_argument("--output_folder", type=str, default="../data")
    parser.add_argument("--job_name", type=str, default=None, help="Job Name. Get from the script.")
    parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp for the job. Also used as the random seed.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser.parse_args()

args = get_args()
print(f"Instruction Generation Manager. Arguments: {args}") # For logging

if args.total_prompts is None:
    if args.repeat is None:
        raise ValueError("Either total prompts or repeat should be specified.")
    args.total_prompts = args.repeat * args.n
else:
    # If total prompts is specified, repeat will be ignored
    args.repeat = int(np.ceil(args.total_prompts / args.n))

# Set the random seed for NumPy
if args.seed is not None:
    np.random.seed(args.seed)
    # Set the random seed for PyTorch
    torch.manual_seed(args.seed)
    # If you are using CUDA (i.e., a GPU), also set the seed for it
    torch.cuda.manual_seed_all(args.seed)

# Create output file / folder
# output_filename = f"Magpie_{args.model_path.split('/')[-1]}_{args.total_prompts}_{args.timestamp}_ins.json"
output_filename = f"magpie_inst-{args.model_path.split('/')[-1]}-{args.total_prompts}-{args.timestamp}.json"
if not args.job_name:
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    output_dir = f"{args.output_folder}/{output_filename}"
else:
    output_dir = f"{args.output_folder}/{args.job_name}/{output_filename}"

# Set the device
# os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# Set generation engine
if args.engine == "vllm":
    # Create vllm instance  
    llm = LLM(model=args.model_path, 
            dtype=args.dtype,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            swap_space=args.swap_space,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed if args.seed is not None else args.timestamp,
            enable_prefix_caching=True)
elif args.engine == "hf":
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={'':torch.cuda.current_device()},
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    )

# todo: tmp fix
# args.model_path = re.sub("/projects/bhuang/models/llm/pretrained/", "", args.model_path)
# args.model_path = re.sub("/lustre/fswork/projects/rech/gkb/commun/models/pretrained/", "", args.model_path)
args.model_path = re.search(r"([^/]+/[^/]+)$", args.model_path).group(1)

# Obtain config from configs/model_configs.json
with open("../configs/model_configs_c.json", "r", encoding="utf-8") as f:
    model_configs = json.load(f)
    model_config = model_configs[args.model_path]
    if args.control_tasks:
        pre_query_template = model_config[f"pre_query_template_{args.control_tasks}"]
        print("Control task: {args.control_tasks}")
    elif args.system_prompt:
        pre_query_template = model_config["pre_query_template_with_system_prompt"]
        print("System prompt enabled. Warning: The system prompt may degrade the performance.")
    else:
        pre_query_template = model_config["pre_query_template"]
    stop_tokens = model_config["stop_tokens"]
    stop_tokens_assistant = model_config["stop_tokens_assistant"]
    stop_tokens += stop_tokens_assistant
    stop_token_ids = model_config["stop_token_ids"]

    # Process early stopping. We found that sometimes LLM will generate responses immediately after the \n token.
    if args.early_stopping:
        stop_tokens.append("\n")

    print(f"Pre-query template: {pre_query_template}")
    print(f"Stop tokens: {stop_tokens}")
    print(f"Stop token ids: {stop_token_ids}")


# Initialize logits processors for llama-3.1
def de_md_logits_processor_for_llama3_1(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        logits[567] = -9999.999 # "##": 567,

    return logits

if args.logits_processor and "llama-3.1" in args.model_path.lower():
    logits_processor = de_md_logits_processor_for_llama3_1
    print(f"Logits processor applied: {logits_processor}")
else:
    logits_processor = None

# Define sampling parameters
sampling_params = SamplingParams(
    n=args.n,
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
    skip_special_tokens=args.skip_special_tokens,
    stop=stop_tokens,
    stop_token_ids=stop_token_ids,
    logits_processors=[logits_processor] if logits_processor else None
)

system_prompt_tasks = [
    (("Information seeking", INFORMATION_SEEKING_PROMPT), 0.05),
    (("Reasoning", REASONING_PROMPT), 0.125),
    (("Planning", PLANNING_PROMPT), 0.05),
    (("Editing", EDITING_PROMPT), 0.10),
    (("Coding & Debugging", CODING_DEBUGGING_PROMPT), 0.125),
    (("Math", MATH_SYSTEM_PROMPT), 0.125),
    (("Role playing", ROLE_PLAYING_PROMPT), 0.10),
    (("Data analysis", DATA_ANALYSIS_PROMPT), 0.125),
    (("Creative writing", CREATIVE_WRITING_PROMPT), 0.10),
    (("Advice seeking", ADVICE_SEEKING_PROMPT), 0.05),
    (("Brainstorming", BRAINSTORMING_PROMPT), 0.05),
]

system_prompt_tasks, system_prompt_task_probs = zip(*system_prompt_tasks)

boost_randomness_probability = 0.5
system_prompt_randomness_boosters = [
    # ("Complex", " Les questions posées peuvent également être complexes, nuancées ou multidimensionnelles."),
    # ("Creative", " Les questions posées peuvent également être créatives et variées."),
    # ("Smart", " Les questions posées peuvent également être astucieuses et originales."),
    # ("Surprising", " Les questions posées peuvent également être inattendues ou celles auxquelles les utilisateurs ne pensent pas immédiatement."),
    # ("Detailed", " Les questions posées peuvent également être détaillées et contextualisées de manière approfondie."),
    ("Complex", "\n\nLes questions posées doivent également être complexes, nuancées ou multidimensionnelles."),
    ("Creative", "\n\nLes questions posées doivent également être créatives et variées."),
    ("Smart", "\n\nLes questions posées doivent également être astucieuses et originales."),
    ("Surprising", "\n\nLes questions posées doivent également être inattendues ou celles auxquelles les utilisateurs ne pensent pas immédiatement."),
    ("Detailed", "\n\nLes questions posées doivent également être détaillées et contextualisées de manière approfondie."),
]

################
# Generate outputs
################
results = []
for rounds in tqdm(range(args.repeat)):

    task_category = None
    task_prompt = None
    randomness_booster_category = None
    if args.system_prompt:
        task_category, task_prompt = random.choices(system_prompt_tasks, system_prompt_task_probs, k=1)[0]
        tmp_task_prompt = task_prompt
        if random.random() < boost_randomness_probability:
            randomness_booster_category, randomness_booster_prompt = random.choice(system_prompt_randomness_boosters)
            tmp_task_prompt += randomness_booster_prompt
        query_template = re.sub("###PLACE_HOLDER###", tmp_task_prompt, pre_query_template)
    else:
        query_template = pre_query_template

    # Generate outputs
    if args.engine == "vllm":
        output = llm.generate(query_template, sampling_params)
        output_list = output[0].outputs
        if args.shuffle:
            random.shuffle(output_list)

    elif args.engine == "hf":
        input = tokenizer.encode(query_template, add_special_tokens=False, return_tensors="pt").to(torch.cuda.current_device())
        # Gemma-2 bug, so we cannot set num_return_sequences > 1. 
        # Instead, we repeat the input n times.
        inputs = input.repeat(args.n, 1).to(torch.cuda.current_device())
        output = model.generate(inputs,
                                tokenizer=tokenizer, 
                                do_sample=True, 
                                temperature=args.temperature, 
                                top_p=args.top_p, 
                                max_length=args.max_tokens, 
                                num_return_sequences=1,
                                )
        # Remove the input from the output
        output_list = tokenizer.batch_decode(output[i][len(inputs[0]):] for i in range(args.n))
        # Stop on the first stop token
        for i, completion in enumerate(output_list):
            for stop_token in stop_tokens:
                if stop_token in completion:
                    output_list[i] = completion[:completion.index(stop_token)]

    # Save outputs
    for i, completion in enumerate(output_list):
        if args.engine == "vllm":
            instruction = completion.text.strip()
        elif args.engine == "hf":
            instruction = completion.strip()

        if args.sanitize:
            sanitized_instruction, class_num = str_utils.instruction_post_process(instruction, args.model_path)
            result = {
                "id": rounds * args.n + i,
                "task_category": task_category,
                "task_prompt": task_prompt,
                "randomness_booster_category": randomness_booster_category,
                "pre_query_template": f"{query_template}",
                "raw_instruction": instruction,
                "instruction": sanitized_instruction,
                "instruction_sanitize_class_num": class_num,
                "response": None,
                "created": int(time.time()),
                "gen_instruction_configs": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "input_generator": f"{args.model_path}",
                    "seed": args.seed,
                },
                "gen_response_configs": None,
            }
        else:
            result = {
                "id": rounds * args.n + i,
                "task_category": task_category,
                "task_prompt": task_prompt,
                "randomness_booster_category": randomness_booster_category,
                "pre_query_template": f"{query_template}",
                "raw_instruction": instruction,
                "response": None,
                "created": int(time.time()),
                "gen_instruction_configs": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "input_generator": f"{args.model_path}",
                    "seed": args.seed,
                },
                "gen_response_configs": None,
            }
        results.append(result)

    # Save the checkpoints every args.checkpoint_every rounds
    if rounds % args.checkpoint_every == 0:
        # with open(output_dir, "w", encoding="utf-8") as f:
        #     json.dump(results, f, indent=4, ensure_ascii=False)
        jsonl_dump(results, output_dir)
        print(f"Checkpoint saved. Total prompts: {len(results)}")

# Save the final results
# with open(output_dir, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=4, ensure_ascii=False)
jsonl_dump(results, output_dir)

print(f"Instruction generated from {args.model_path}. Total prompts: {len(results)}")
