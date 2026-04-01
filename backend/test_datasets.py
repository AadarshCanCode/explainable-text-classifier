from datasets import load_dataset
try:
    ds = load_dataset("tasksource/jigsaw_toxicity", split="train[:10]")
    t = ds['comment_text']
    print(type(t))
except Exception as e:
    print("Error:", e)
