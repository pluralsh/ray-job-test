import ray
from ray import tune

# 1. Define an objective function.
def objective(config):
    score = config["a"] ** 2 + config["b"]
    return {"score": score}


resource_group = tune.PlacementGroupFactory(
    [{"CPU": 1, "memory": 1024*1024*1024}],
    strategy="SPREAD",
)
trainable_with_resources = tune.with_resources(objective, resource_group)
#trainable_with_resources = objective
ray.init(
    address="auto",
    _redis_password="5241590000000000",
    runtime_env={"env_vars": {"TUNE_MAX_PENDING_TRIALS_PG": "100"}},
    log_to_driver=False,
)


ray.autoscaler.sdk.request_resources(bundles=[{"CPU": 3, "memory": 15*1024*1024*1024}])

# 2. Define a search space.
search_space = {
    "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    "b": tune.choice([1, 2, 3]),
}

# 3. Start a Tune run and print the best result.
tuner = tune.Tuner(trainable_with_resources, param_space=search_space)
results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)
