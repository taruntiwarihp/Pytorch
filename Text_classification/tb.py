from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

event_acc = EventAccumulator('logs/tensorboard_2022-04-01-05-18_robert/events.out.tfevents.1648790350.instance-1')
event_acc.Reload()

tags = event_acc.Tags()["scalars"]

for tag in tags:
    event_list = event_acc.Scalars(tag)

    values = list(map(lambda x: x.value, event_list))
    step = list(map(lambda x: x.step, event_list))
    # r = {"metric": [tag] * len(step), "value": values, "step": step}
    r = {"step": step, "value": values}
    r = pd.DataFrame(r)

    r.to_csv('TB/new_data/{}.csv'.format(tag))


print(len(tags))