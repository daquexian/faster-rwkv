import os
import sys
import time
from collections import defaultdict

import fasterrwkv as fr

USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant:"
MAX_OUTPUT_LENGTH = int(os.getenv("FR_MAX_OUTPUT_LEN", 999))
END_OF_SENTENCE = 0
DOUBLE_NEW_LINE = "\n\n"
TOP_P = 0.0
PRESENCE_PENALTY = 0.8
FREQUENCY_PENALTY = 0.8
PENALTY_DECAY = 0.996
GLOBAL_PENALTY = os.getenv("FR_GLOBAL_PENALTY", False)
SHOW_SPEED = os.getenv("FR_SHOW_SPEED", False)

tokenizer = fr.Tokenizer(sys.argv[1])
sampler = fr.Sampler()
model = fr.Model(sys.argv[2], sys.argv[3])
if len(sys.argv) == 5:
    model.load_state_file(sys.argv[4])
occurences = defaultdict(float)
while True:
    question = input(USER_PREFIX)
    print(ASSISTANT_PREFIX, end="")
    prompt = USER_PREFIX + question + DOUBLE_NEW_LINE + ASSISTANT_PREFIX
    start = time.time()
    prompt_ids = tokenizer.encode(prompt)
    encode_time = time.time() - start
    start = time.time()
    output = model(prompt_ids)
    response = ""
    for num_new_tokens in range(MAX_OUTPUT_LENGTH):
        for id, occurence in occurences.items():
            output[id] -= FREQUENCY_PENALTY * occurence + PRESENCE_PENALTY
            occurence *= PENALTY_DECAY
        output[END_OF_SENTENCE] = -1e30
        output_id = sampler.sample(output, 1.0, 1, TOP_P)
        occurences[output_id] += 1
        output_str = tokenizer.decode(output_id)
        print(output_str, end="")
        response += output_str
        # it is important to pass the stop word (DOUBLE_NEW_LINE) to the model,
        # or it will stop incorrectly in the next iteration.
        output = model(output_id)
        if response.endswith(DOUBLE_NEW_LINE):
            break
    total_time = time.time() - start
    if SHOW_SPEED:
        total_time *= 1000
        print("-- time: ", total_time, "ms")
        print("-- num tokens: ", len(prompt_ids) + num_new_tokens)
        print("-- ms per token: ", total_time / (len(prompt_ids) + num_new_tokens))
        print("-- tokens per second: ", (len(prompt_ids) + num_new_tokens) / total_time * 1000)
        print()
    if not GLOBAL_PENALTY:
        occurences.clear()

