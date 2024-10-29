#!/usr/bin/env python3
import argparse
import datetime
import os
from glob import glob

import numpy as np
import pandas as pd
import plotly.express as px
from evalica import bradley_terry, Winner, pairwise_frame
from scipy.special import expit
from tqdm import tqdm

from utils import load_questions, load_model_answers

from rich.console import Console
from rich.table import Table

BASELINE_MODEL_NAME = "gpt-3.5-turbo-0125"


def compute_ratings(df: pd.DataFrame, initial: float = 1000., base: float = 10.,
                    scale: float = 400.) -> 'pd.Series[str]':
    df = df.copy()

    df['winner'] = df['winner'].map({
        'model_a': Winner.X,
        'model_b': Winner.Y,
        'tie': Winner.Draw,
        'tie (bothbad)': Winner.Draw,
    })

    result = bradley_terry(
        df['model_a'],
        df['model_b'],
        df['winner'],
        weights=df['answer_len_delta'] * 2,
        tolerance=1e-8
    )

    scores = initial + np.log(result.scores) / np.log(base) * scale

    if BASELINE_MODEL_NAME in scores.index:
        scores += initial - scores[BASELINE_MODEL_NAME]

    return scores.sort_values(ascending=False, kind="stable")


def get_bootstrap_result(battles, func_compute_ratings, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_ratings(battles.sample(frac=1.0, replace=True, random_state=i)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def pretty_print_two_ratings(ratings_1, ratings_2, column_names):
    df = pd.DataFrame([
        [n, ratings_1[n], ratings_2[n]] for n in ratings_1.keys()
    ], columns=["Model", column_names[0], column_names[1]]).sort_values(column_names[0], ascending=False).reset_index(
        drop=True)
    df[column_names[0]] = (df[column_names[0]] + 0.5).astype(int)
    df[column_names[1]] = (df[column_names[1]] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def visualize_bootstrap_scores(df, title):
    bars = pd.DataFrame(dict(
        lower=df.quantile(.025),
        rating=df.quantile(.5),
        upper=df.quantile(.975))
    ).reset_index(names="model").sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    fig = px.scatter(bars, x="model", y="rating", error_y="error_y",
                     error_y_minus="error_y_minus", text="rating_rounded",
                     title=title)
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating",
                      height=600)
    return fig


def predict_win_rate(ratings: dict[str, float], scale: float = 400., base: float = 10.) -> pd.DataFrame:
    scores = pd.Series(ratings).sort_index()
    scores /= scale
    scores = base ** scores

    df = pairwise_frame(scores)
    df.index.name = "model_b"
    df.columns = df.index.copy(name="model_a")
    np.fill_diagonal(df.values, np.nan)

    return df


def get_win_rate_column(df, column, baseline=BASELINE_MODEL_NAME):
    to_dict = df[["model", column]].set_index("model").to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table[baseline].fillna(0.5).apply(lambda x: round(x * 100, 2))


def get_battles_from_judgment(judge_name, answers_lengths, first_game_only=False, WEIGHT=3, length_controlled=False):
    arena_hard_battles = pd.DataFrame()

    print("Turning judgment results into battles...")

    directory = f"data/arena-hard-v0.1/model_judgment/{judge_name}"
    assert os.path.exists(directory)
    for file in tqdm(glob(f"{directory}/*jsonl")):
        df = pd.read_json(file, lines=True)

        for _, row in df.iterrows():
            if length_controlled:
                _model_name = row["model"].split('/')[-1]
                answers_length_deltas = (answers_lengths.loc[BASELINE_MODEL_NAME] - answers_lengths.loc[_model_name])
                answer_length_delta = (answers_lengths.loc[BASELINE_MODEL_NAME][row["question_id"]] -
                                       answers_lengths.loc[_model_name][row["question_id"]])
                normalized_answer_delta_weight = expit(answer_length_delta / answers_length_deltas.std())
            else:
                normalized_answer_delta_weight = 0.5

            # game 1
            output = {
                "question_id": row["question_id"],
                "model_a": BASELINE_MODEL_NAME,
                "model_b": row["model"],
                "answer_len_delta": 0.5
            }

            game = row["games"][0]

            weight = 1
            if game["score"] == "A=B":
                output["winner"] = "tie"
            elif game["score"] == "A>B":
                output["winner"] = "model_a"
            elif game["score"] == "A>>B":
                output["winner"] = "model_a"
                weight = WEIGHT
            elif game["score"] == "B>A":
                output["winner"] = "model_b"
                output['answer_len_delta'] = normalized_answer_delta_weight
            elif game["score"] == "B>>A":
                output["winner"] = "model_b"
                output['answer_len_delta'] = normalized_answer_delta_weight
                weight = WEIGHT
            else:
                weight = 0

            if weight:
                arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])

            if not first_game_only:
                # game 2
                output = {
                    "question_id": row["question_id"],
                    "model_a": BASELINE_MODEL_NAME,
                    "model_b": row["model"],
                    "answer_len_delta": 0.5
                }

                game = row["games"][1]

                weight = 1
                if game["score"] == "A=B":
                    output["winner"] = "tie"
                elif game["score"] == "A>B":
                    output["winner"] = "model_b"
                    output['answer_len_delta'] = normalized_answer_delta_weight
                elif game["score"] == "A>>B":
                    output["winner"] = "model_b"
                    output['answer_len_delta'] = normalized_answer_delta_weight
                    weight = WEIGHT
                elif game["score"] == "B>A":
                    output["winner"] = "model_a"
                elif game["score"] == "B>>A":
                    output["winner"] = "model_a"
                    weight = WEIGHT
                else:
                    weight = 0

                if weight:
                    arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])
    arena_hard_battles.to_json("data/arena_hard_battles.jsonl", lines=True, orient="records")
    return arena_hard_battles


def get_models_answers_lengths(questions_df, model_answers_df) -> pd.DataFrame:
    model_answers_lengths = []
    for model_name, row in model_answers_df.iterrows():
        model_stats = {'model_name': model_name}
        for question in questions_df.index:
            if question in row and isinstance(row[question], dict):
                turn = row[question]["choices"][0]["turns"][0]
                model_stats[question] = turn["token_len"]
            else:
                model_stats[question] = 0
        model_answers_lengths.append(model_stats)
    return pd.DataFrame(model_answers_lengths).set_index('model_name')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="arena-hard-v0.1")
    parser.add_argument("--judge-name", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--baseline", type=str, default=BASELINE_MODEL_NAME)
    parser.add_argument("--load-battles", action="store_true")
    parser.add_argument("--load-bootstrap", action="store_true")
    parser.add_argument("--show-elo", action="store_true")
    parser.add_argument("--length-control", action="store_true")
    parser.add_argument("--weight", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--output", action="store_true")
    parser.add_argument("--first-game-only", action="store_true")
    args = parser.parse_args()
    print(args)
    assert not args.load_bootstrap or (
            args.load_battles and args.load_bootstrap), "If loading prexisting bootstrapping data, you must also load preexisting battles."

    question_file = os.path.join("data", args.bench_name, "question.jsonl")
    questions_df = pd.DataFrame(load_questions(question_file)).set_index('question_id')

    answer_dir = os.path.join("data", args.bench_name, "model_answer")
    model_answers_df = pd.DataFrame(load_model_answers(answer_dir)).T

    models_answers_lengths = get_models_answers_lengths(questions_df, model_answers_df)

    if args.load_battles:
        assert os.path.exists("data/arena_hard_battles.jsonl")
        battles = pd.read_json("data/arena_hard_battles.jsonl", lines=True)
    else:
        battles = get_battles_from_judgment(args.judge_name, models_answers_lengths, args.first_game_only, args.weight,
                                            args.length_control)

    bootstrap_ratings = compute_ratings(battles)

    models_names = bootstrap_ratings.index

    if args.load_bootstrap:
        bootstrap_ratings_lu = pd.read_json("data/bootstrapping_results.jsonl", lines=True)
    else:
        bootstrap_ratings_lu = get_bootstrap_result(battles, compute_ratings, args.num_rounds)
        bootstrap_ratings_lu.to_json("data/bootstrapping_results.jsonl", lines=True, orient="records")

    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats['results'].astype('object')

    for i, model in enumerate(models_names):
        assert model in bootstrap_ratings_lu.columns

        stats.at[i, "model"] = model
        stats.at[i, "score"] = bootstrap_ratings[model]
        stats.at[i, "lower"] = np.percentile(bootstrap_ratings_lu[model], 2.5)
        stats.at[i, "upper"] = np.percentile(bootstrap_ratings_lu[model], 97.5)

        stats.at[i, "avg_tokens"] = models_answers_lengths.loc[model.split('/')[-1]].mean()
        stats.at[i, "std_tokens"] = models_answers_lengths.loc[model.split('/')[-1]].std()

        stats.at[i, "results"] = bootstrap_ratings_lu[model].tolist()

    if not args.show_elo:
        stats.sort_values(by="model", inplace=True)
        stats["score"] = get_win_rate_column(stats, "score", args.baseline).tolist()
        stats["lc_score"] = get_win_rate_column(stats, "score", args.baseline).tolist()
        stats["lower"] = get_win_rate_column(stats, "lower", args.baseline).tolist()
        stats["upper"] = get_win_rate_column(stats, "upper", args.baseline).tolist()
        decimal = 1
    else:
        decimal = 0
        stats = stats.astype({"score": int, "lower": int, "upper": int})

    # Initialize Rich Console
    console = Console()
    
    # Create a Rich Table
    table = Table(show_header=True, header_style="bold magenta", show_lines=False)
    table.add_column("Model", width=40)
    table.add_column("Score", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Avg. #Tokens", justify="right")
    
    # Sort values by 'score' as per your existing code
    stats.sort_values(by="score", ascending=False, inplace=True)
    
    # Add rows to the table
    for _, row in stats.iterrows():
        interval = f"({round(row['lower'] - row['score'], decimal)}, {round(row['upper'] - row['score'], decimal)})"
        table.add_row(
            row['model'],
            f"{round(row['score'], decimal)}",
            interval,
            f"{int(row['avg_tokens'])}"
        )
    
    # Print the table using Rich
    console.print(table)

    if args.output:
        cur_date = datetime.datetime.now()
        date_str = cur_date.strftime("%Y%m%d")
        stats.to_json(f"arena_hard_leaderboard_{date_str}.json", orient="records", indent=4)
