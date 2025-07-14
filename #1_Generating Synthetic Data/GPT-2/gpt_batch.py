from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

# Loading model and tokenizer
model = AutoModelForCausalLM.from_pretrained("linuxLogsModel_10K")
tokenizer = AutoTokenizer.from_pretrained("linuxLogsModel_10K")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id
)

# Prompt template
# prompt = "[06:13:02.227914980] (+0.000000334) mendax syscall_exit_fcntl: { cpu_id = 2 }, { procname = \"elasticsearch\", pid = 11822, tid = 11859 }, { ret = 2, arg = 50 }\n"

prompt = """Generate realistic LTTng Linux kernel logs in the following format:
[timestamp] (+delta) hostname syscall_event: { cpu_id = X }, { procname = "name", pid = N, tid = M }, { ret/arg fields }

Example output:
[06:13:02.227908688] (+0.000003700) mendax syscall_entry_accept: { cpu_id = 2 }, { procname = "elasticsearch", pid = 11822, tid = 11859 }, { fd = 553, upeer_addrlen = 246916502706640 }
[06:13:02.227912438] (+0.000003750) mendax syscall_exit_accept: { cpu_id = 2 }, { procname = "elasticsearch", pid = 11822, tid = 11859 }, { ret = 576, upeer_sockaddr = 246916502706648, upeer_addrlen = 246916502706640 }
"""

num_total = 10000
batch_size = 200 
temperature=1.5
top_p=0.9

csv_path = "synthetic_linux_logs_10K.csv"
columns = ["log_entry"]

with open(csv_path, "w") as f:
    f.write(",".join(columns) + "\n") 

# Batch generation loop
for batch_start in tqdm(range(0, num_total, batch_size), desc="Generating logs"):
    outputs = generator(
        prompt,
        max_new_tokens=80,
        num_return_sequences=batch_size,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )

    synthetic_logs = []
    for output in outputs:
        text = output['generated_text'][len(prompt):].strip()
        text = text.split("\n")[0]
        synthetic_logs.append(text)

    # CSV Output
    df = pd.DataFrame({"log_entry": synthetic_logs})
    df.to_csv(csv_path, mode="a", header=False, index=False)

print(f"Exported 10K logs to {csv_path}")
