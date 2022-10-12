import ray
import os
import requests
ray.init()
# ray.autoscaler.sdk.request_resources(num_cpus=2)
@ray.remote
class Counter:
    def __init__(self):
        # Used to verify runtimeEnv
        self.name = os.getenv("counter_name")
        self.counter = 0
    def inc(self):
        self.counter += 1
    def get_counter(self):
        return "{} got {}".format(self.name, self.counter)
counter = Counter.remote()
for _ in range(5):
    ray.get(counter.inc.remote())
    print(ray.get(counter.get_counter.remote()))
print(requests.__version__)
