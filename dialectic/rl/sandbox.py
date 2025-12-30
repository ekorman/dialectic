def train_pg(
    env,
    generate,
    net,  # or something like lobprobs?
    answer_extractor,
    reward_fn,  # should take in extracted answer and full message
    max_episodes,
):
    for _ in range(max_episodes):
        pass
