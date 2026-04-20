import argparse
import json

import pandas as pd
from mlflow.tracking import MlflowClient


def serialize_runs_with_relationships(runs):
	"""Serialize MLflow runs and derive parent/child relationships from tags."""
	serialized_runs = []
	for run in runs:
		tags = dict(run.data.tags)
		serialized_runs.append(
			{
				"run_id": run.info.run_id,
				"run_name": tags.get("mlflow.runName"),
				"experiment_id": run.info.experiment_id,
				"status": run.info.status,
				"lifecycle_stage": run.info.lifecycle_stage,
				"start_time": run.info.start_time,
				"end_time": run.info.end_time,
				"artifact_uri": run.info.artifact_uri,
				"parent_run_id": tags.get("mlflow.parentRunId"),
				"metrics": dict(run.data.metrics),
				"params": dict(run.data.params),
				"tags": tags,
			}
		)

	child_run_ids_by_parent = {}
	for run in serialized_runs:
		parent_run_id = run["parent_run_id"]
		if parent_run_id:
			child_run_ids_by_parent.setdefault(parent_run_id, []).append(run["run_id"])

	for run in serialized_runs:
		child_run_ids = child_run_ids_by_parent.get(run["run_id"], [])
		run["child_run_ids"] = child_run_ids
		run["child_run_count"] = len(child_run_ids)
		run["is_child_run"] = run["parent_run_id"] is not None
		run["is_parent_run"] = len(child_run_ids) > 0

	return serialized_runs


def get_all_runs(experiment_name=None, max_results=1000, only_finished=True):
	"""Return MLflow runs for one experiment or for all experiments."""
	client = MlflowClient()

	if experiment_name:
		experiment = client.get_experiment_by_name(experiment_name)
		if experiment is None:
			raise ValueError(f"Experiment '{experiment_name}' not found.")
		experiment_ids = [experiment.experiment_id]
	else:
		experiment_ids = [
			experiment.experiment_id
			for experiment in client.search_experiments()
		]

	if not experiment_ids:
		return []

	filter_string = "attributes.status = 'FINISHED'" if only_finished else ""
	runs = client.search_runs(
		experiment_ids=experiment_ids,
		filter_string=filter_string,
		max_results=max_results,
		order_by=["attributes.start_time DESC"],
	)

	return serialize_runs_with_relationships(runs)


def runs_to_dataframe(runs):
	"""Convert serialized MLflow runs to a flattened pandas DataFrame."""
	flattened_runs = []
	for run in runs:
		base_row = {
			"run_id": run["run_id"],
			"run_name": run["run_name"],
			"experiment_id": run["experiment_id"],
			"status": run["status"],
			"lifecycle_stage": run["lifecycle_stage"],
			"start_time": run["start_time"],
			"end_time": run["end_time"],
			"artifact_uri": run["artifact_uri"],
			"parent_run_id": run["parent_run_id"],
			"child_run_count": run["child_run_count"],
			"child_run_ids": json.dumps(run["child_run_ids"]),
			"is_child_run": run["is_child_run"],
			"is_parent_run": run["is_parent_run"],
		}

		for key, value in run["metrics"].items():
			base_row[f"metric.{key}"] = value
		for key, value in run["params"].items():
			base_row[f"param.{key}"] = value
		for key, value in run["tags"].items():
			base_row[f"tag.{key}"] = value

		flattened_runs.append(base_row)

	return pd.DataFrame(flattened_runs)


def export_runs_to_csv(runs, csv_path):
	"""Export serialized MLflow runs to a CSV file."""
	dataframe = runs_to_dataframe(runs)
	dataframe.to_csv(csv_path, index=False)
	return dataframe


def main(experiment_name=None, max_results=1000, as_json=False, only_finished=True, csv_path=None, return_dataframe=False):
	runs = get_all_runs(
		experiment_name=experiment_name,
		max_results=max_results,
		only_finished=only_finished,
	)

	if csv_path:
		dataframe = export_runs_to_csv(runs, csv_path)
		print(f"Exported {len(dataframe)} runs to {csv_path}")
		if return_dataframe:
			return dataframe
		return runs

	if return_dataframe:
		return runs_to_dataframe(runs)

	if as_json:
		print(json.dumps(runs, indent=2))
		return runs

	for run in runs:
		print(
			f"{run['run_id']} | {run['run_name']} | "
			f"experiment={run['experiment_id']} | status={run['status']}"
		)

	return runs


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--experiment_name",
		type=str,
		default=None,
		help="Optional MLflow experiment name to restrict the search.",
	)
	parser.add_argument(
		"--max_results",
		type=int,
		default=1000,
		help="Maximum number of runs to return.",
	)
	parser.add_argument(
		"--include_unfinished",
		action="store_true",
		help="Include non-finished runs.",
	)
	parser.add_argument(
		"--csv_path",
		type=str,
		default=None,
		help="Optional path to export the runs as CSV.",
	)
	parser.add_argument(
		"--json",
		action="store_true",
		help="Print runs as JSON instead of a compact text summary.",
	)
	args = parser.parse_args()
	main(
		experiment_name=args.experiment_name,
		max_results=args.max_results,
		as_json=args.json,
		only_finished=not args.include_unfinished,
		csv_path=args.csv_path,
	)
