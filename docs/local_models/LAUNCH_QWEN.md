# Running Qwen3.8-27B on camel

camel serves one model at a time on its 32 GB GPU. Two versions of Qwen3.8-27B are installed:
**q5** (the default) and **q6**. Start the server once; it keeps running after you log out.

## Start and stop

```bash
~/edwin/llama_server.sh                       # q5, one user, 196,608 tokens of context
~/edwin/llama_server.sh start --model q6      # q6, for hard coding sessions
~/edwin/llama_server.sh start --slots 4       # four people share the server
~/edwin/llama_server.sh start --kv q8_0 --ctx 229376   # q5 with the longest context
~/edwin/llama_server.sh status                # is it running, and how much GPU it uses
~/edwin/llama_server.sh stop                  # frees the GPU; stop before switching model
```

## Which setting to use

| Setting | Context | GPU memory (of 32,768 MiB) | Speed, one user | Speed, each of four users | Use it for |
|---|---|---|---|---|---|
| q5 (default) | 196,608 | 31,826 MiB | 28 tokens/s | about 15.5 tokens/s | chat, summaries, most coding |
| q5 `--kv q8_0 --ctx 229376` | 229,376 | 28,046 MiB | 28 tokens/s | about 15.5 tokens/s | very long documents |
| q6 | 196,608 | 31,322 MiB | 24 tokens/s | about 14.5 tokens/s | hard coding tasks |

Memory and one-user speed come from `results/camel_settings.csv` in the Qwen3Benchmark repository
(q6 measured with four slots); four-user speeds from `results/camel_speed.csv`.

With `--slots 4`, the context is split evenly: 196,608 tokens become 49,152 per person.

## Use it

**Web chat from your laptop.** Open a tunnel, then open the page in your browser:

```bash
ssh -L 8080:127.0.0.1:8080 camel              # camel = your ssh alias for this machine
```

Then go to http://localhost:8080. Chats stay in your browser.

**Coding agent on camel.** Open Qwen Code in the project folder:

```bash
cd <project> && ~/edwin/qwen                  # new session
~/edwin/qwen -c                               # continue the last session in this folder
~/edwin/qwen -r                               # choose an earlier session
```

Inside Qwen Code, `/resume` switches session and `/clear` starts a fresh one.

## For complex code

Both versions fail when the task is described only by its goal. Give the model the interface and
the rules it must follow, let it run the code, and check the result against numbers you already
know. In our tests, both versions wrote a file this way that passed the repository's tests. In one
reply without running the code, q6 still passed and q5 failed.

The full comparison of models and settings is in `docs/camel_model_bench.md` in the Qwen3Benchmark repository.
